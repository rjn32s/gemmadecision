"""Bounded real HTTP serving check. Run on a CUDA host, never as a CI unit test.

Starts only its own process group and shuts it down even on failure. No request
or failure is retried. These hand-authored inputs measure serving mechanics,
not model accuracy, and use distinct case IDs to avoid score deduplication.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import secrets
import signal
import subprocess
import sys
import time
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from gemmadecision import AsyncDecisionClient, Choice, Noul, Score


OPTIONS = {
    "cards": "Help with lost, stolen, frozen or replaced payment cards.",
    "transfers": "Help with bank transfers and recipient payments.",
    "cash": "Help with ATM withdrawals and cash deposits.",
    "other": "Help with another banking issue.",
}
STATES = [
    "My bank card went missing yesterday. How can I replace it?",
    "I sent a transfer to the wrong recipient and need help.",
    "The ATM kept my cash but did not update my balance.",
    "I need to update the postal address on my account.",
    "I froze my payment card and now need to unfreeze it.",
    "An incoming bank transfer has not arrived yet.",
    "The cash machine charged me but dispensed no money.",
    "Where can I download my monthly account statement?",
]

SIMPLE_SMOKE = r'''
import json
import time
from typing import Literal
from pydantic_ai import Agent
from gemmadecision import decide, rank, GemmaDecisionModel
from gemmadecision.simple import get_engine
choices = ["billing", "technical"]
started = time.perf_counter()
first = decide("I was charged twice for the same order.", choices=choices)
first_elapsed = time.perf_counter() - started
assert isinstance(first, str) and first in choices
started = time.perf_counter()
second = decide("The application will not start on my computer.", choices=choices)
warm_elapsed = time.perf_counter() - started
assert isinstance(second, str) and second in choices
detailed = rank("I was charged twice for the same order.", choices=choices)
assert len(detailed.ranked) == len(choices)
assert {row.text for row in detailed.ranked} == set(choices)
shared_engine = get_engine()
local_agent = Agent(
    GemmaDecisionModel.local(), output_type=Literal["billing", "technical"],
    instructions="Which department should handle this ticket?",
)
started = time.perf_counter()
typed = local_agent.run_sync("I was charged twice for the same order.")
typed_elapsed = time.perf_counter() - started
assert typed.output in choices
assert typed.response.provider_name == "gemmadecision"
assert typed.response.provider_url == "local://gemmadecision"
assert get_engine() is shared_engine
print("GEMMADECISION_SIMPLE_RESULT=" + json.dumps({
    "passed": True, "first_result": first, "second_result": second,
    "first_call_including_load_s": first_elapsed, "warm_decide_call_s": warm_elapsed,
    "ranking": detailed.model_dump(mode="json"), "engine": get_engine().metadata(),
    "native_pydantic_ai_local": {
        "passed": True, "output": typed.output, "warm_agent_call_s": typed_elapsed,
        "provider_url": typed.response.provider_url,
        "provider_details": typed.response.provider_details,
        "shared_engine_identity_unchanged": True,
    },
    "accuracy_evaluated": False,
}), flush=True)
'''


class TypedDecision(BaseModel):
    route: Literal["cards", "transfers", "cash", "other"] = Field(
        description="The department best suited to the customer's request."
    )
    urgent: bool = Field(description="The customer needs immediate action to prevent financial loss.")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def write_report(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def runtime_versions() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("gemmadecision", "granian", "torch", "transformers", "vllm", "pydantic-ai-slim", "httpx"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        result["gpu_inventory"] = gpu.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        result["gpu_inventory"] = None
    return result


async def wait_ready(base_url: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.perf_counter() + timeout
    async with httpx.AsyncClient(base_url=base_url, timeout=1) as http:
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited before readiness with code {process.returncode}")
            try:
                response = await http.get("/ready")
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("status") == "ready":
                        return payload
            except (httpx.RequestError, ValueError):
                pass
            await asyncio.sleep(.25)
    raise TimeoutError("Server did not become ready before the startup deadline")


async def check_api(base_url: str, key: str) -> dict:
    from pydantic_ai import Agent
    from gemmadecision.integrations.pydantic_ai import GemmaDecisionModel

    async with AsyncDecisionClient(base_url, api_key=key, timeout=30) as client:
        answer = await client.system_one(STATES[0], {
            "route": Choice(instructions="Choose the correct banking department.", criteria=OPTIONS),
            "urgent": Noul(instructions="The customer needs immediate help to prevent misuse of their card."),
            "priority": Score(instructions="Assess how urgently this request needs attention.",
                              criteria=["Routine request", "Needs attention soon", "Needs immediate attention"]),
        })
        assert set(answer.answers) == {"route", "urgent", "priority"}
        assert answer.answers["route"].type == "choice"
        assert answer.answers["urgent"].type == "noul"
        assert answer.answers["priority"].type == "score"
        assert all(math.isfinite(p) for value in answer.answers.values() for p in value.probabilities.values())

        async with GemmaDecisionModel(client=client) as model:
            agent = Agent(model, output_type=TypedDecision,
                          instructions="Classify the banking request into the provided typed fields.")
            result = await agent.run(STATES[0])
            assert isinstance(result.output, TypedDecision)
            assert result.output.route in OPTIONS and isinstance(result.output.urgent, bool)

        ranked = await client.rank(STATES[1], OPTIONS, question="Choose the best banking department.")
        assert len(ranked.ranked) == 4
        assert {item.candidate for item in ranked.ranked} == set(OPTIONS)

    async with httpx.AsyncClient(base_url=base_url, timeout=30) as http:
        body = {"state": STATES[0], "candidates": OPTIONS, "question": "Choose a department."}
        unauthorized = await http.post("/v1/rank", json=body)
        assert unauthorized.status_code == 401, unauthorized.status_code
        overlong = await http.post("/v1/rank", json={**body, "state": "example " * 4000},
                                   headers={"Authorization": "Bearer " + key})
        assert overlong.status_code == 422, overlong.status_code

    return {
        "all_three_question_types": answer.model_dump(mode="json"),
        "pydantic_ai_output": result.output.model_dump(mode="json"),
        "pydantic_ai_output_type_valid": True,
        "rank_endpoint_valid": True,
        "missing_key_status": unauthorized.status_code,
        "overlong_state_status": overlong.status_code,
        "accuracy_evaluated": False,
    }


async def measure(base_url: str, key: str, concurrency: int, count: int) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    latencies, pairs, tokens, cached = [], [], [], []
    async with AsyncDecisionClient(base_url, api_key=key, timeout=30) as client:
        before = await client.health()

        async def request(index: int) -> None:
            # Include a unique identifier in every request, including across
            # load levels, so engine score deduplication cannot inflate throughput.
            state = STATES[index % len(STATES)] + f" Case reference: HTTP-{concurrency}-{index}."
            async with semaphore:
                started = time.perf_counter()
                result = await client.system_one(state, {
                    "route": Choice(instructions="Choose the correct banking department.", criteria=OPTIONS)
                })
                latencies.append((time.perf_counter() - started) * 1000)
            assert result.answers["route"].choice in OPTIONS
            assert set(result.answers["route"].scores) == set(OPTIONS)
            assert result.usage.cached_pairs == 0
            pairs.append(len(result.answers["route"].scores))
            tokens.append(result.usage.input_tokens)
            cached.append(result.usage.cached_pairs)

        started = time.perf_counter()
        await asyncio.gather(*(request(i) for i in range(count)))
        elapsed = time.perf_counter() - started
        after = await client.health()

    batches = after["model_batches"] - before["model_batches"]
    return {
        "concurrency": concurrency,
        "requests": count,
        "candidate_pairs": sum(pairs),
        "input_tokens_including_repeated_state": sum(tokens),
        "cached_pairs": sum(cached),
        "elapsed_s": elapsed,
        "requests_per_s": count / elapsed,
        "candidate_pairs_per_s": sum(pairs) / elapsed,
        "request_latency_ms_p50": percentile(latencies, .5),
        "request_latency_ms_p95": percentile(latencies, .95),
        "request_latency_ms_min": min(latencies),
        "request_latency_ms_max": max(latencies),
        "model_batcher_calls": batches,
        "requests_per_batcher_call": count / batches if batches else None,
        "completed_requests_delta": after["completed_requests"] - before["completed_requests"],
        "latency_scope": "HTTP SDK call, excluding client semaphore wait; includes server queue and inference",
    }


async def run_checks(args, process, key, report, start) -> None:
    url = "http://127.0.0.1:18700"
    report["health_after_startup"] = await wait_ready(url, process, min(90, args.max_seconds * .6))
    report["startup_to_ready_s"] = time.perf_counter() - start
    report["checks"] = await check_api(url, key)
    print("HTTP authentication, context refusal, all question types, and native PydanticAI passed", flush=True)
    async with AsyncDecisionClient(url, api_key=key, timeout=30) as client:
        for index in range(3):
            await client.decide(STATES[index] + f" Warmup {index}.", OPTIONS)
    report["warmup_requests"] = 3
    for concurrency in (1, 8, 32):
        measured = await measure(url, key, concurrency, args.requests)
        report["measurements"].append(measured)
        write_report(args.output, report)
        print(f"HTTP concurrency={concurrency}: {measured['requests_per_s']:.2f} requests/s; "
              f"p50={measured['request_latency_ms_p50']:.1f}ms p95={measured['request_latency_ms_p95']:.1f}ms", flush=True)
    async with AsyncDecisionClient(url, api_key=key) as client:
        report["health_after_measurement"] = await client.health()


def stop_process_group(process: subprocess.Popen) -> None:
    # start_new_session makes pid the group ID; never target unrelated workers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The CLI parent can exit before Granian's worker. Sweep this same group,
    # even if the parent has already exited, to avoid a leftover GPU allocation.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def check_simple_api(args, remaining_seconds: float) -> dict:
    """A fresh process tests public lazy APIs only after the HTTP worker exits."""
    if remaining_seconds < 5:
        raise TimeoutError("No time remains for the one-call API check")
    environment = dict(os.environ)
    environment["GEMMADECISION_MODEL_PATH"] = str(args.model_path.resolve())
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment.pop("GEMMADECISION_API_KEY", None)
    process = subprocess.Popen(
        [sys.executable, "-c", SIMPLE_SMOKE], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, env=environment, start_new_session=True,
    )
    captured = ""
    try:
        captured, _ = process.communicate(timeout=max(1, remaining_seconds - 2))
        if process.returncode:
            raise RuntimeError(f"One-call API subprocess exited with code {process.returncode}")
        prefix = "GEMMADECISION_SIMPLE_RESULT="
        rows = [line[len(prefix):] for line in captured.splitlines() if line.startswith(prefix)]
        if len(rows) != 1:
            raise RuntimeError("One-call API subprocess did not emit exactly one result")
        return json.loads(rows[0])
    finally:
        stop_process_group(process)
        args.output.with_suffix(".simple.log").write_text(captured)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=["torch", "vllm"], default="torch")
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--requests", type=int, default=32, choices=range(32, 65))
    args = parser.parse_args()
    if args.max_seconds < 60:
        parser.error("--max-seconds must allow at least 60 seconds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Output already exists; choose a new filename to preserve earlier evidence")
    if os.name != "posix":
        parser.error("This cloud benchmark requires a POSIX CUDA host")
    key = secrets.token_urlsafe(32)
    environment = dict(os.environ)
    environment["GEMMADECISION_API_KEY"] = key
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, "-m", "gemmadecision.cli", "serve", "--model-path", str(args.model_path),
               "--device", "cuda", "--port", "18700", "--backend", args.backend,
               "--cache-size", "0", "--batch-wait-ms", "1", "--max-batch-size", "32"]
    report = {
        "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_versions(), "backend": args.backend, "command": command,
        "model_source": str(args.model_path), "cache_size": 0, "batch_wait_ms": 1,
        "max_batch_size": 32, "max_batch_tokens": 8192, "accuracy_evaluated": False,
        "authentication": "random ephemeral bearer key, never written to report",
        "measurements": [], "server_log": args.output.with_suffix(".server.log").name,
        "batching_note": "model_batches counts scheduler calls, not individual encoder forward chunks",
    }
    write_report(args.output, report)
    start = time.perf_counter()
    process = None
    try:
        with args.output.with_suffix(".server.log").open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)

            async def bounded():
                # Reserve time to stop the HTTP worker and perform a fresh
                # in-process one-call API check without two resident models.
                async with asyncio.timeout(args.max_seconds - 35):
                    await run_checks(args, process, key, report, start)

            asyncio.run(bounded())
        stop_process_group(process)
        process = None
        report["simple_api"] = check_simple_api(args, args.max_seconds - (time.perf_counter() - start))
        print("Public one-call decide/rank APIs passed with a fresh offline model load", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {str(error).replace(key, '[redacted]')}"
    finally:
        if process is not None:
            stop_process_group(process)
        report["total_elapsed_s"] = time.perf_counter() - start
        write_report(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "total_elapsed_s": report["total_elapsed_s"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
