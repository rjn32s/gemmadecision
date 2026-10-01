#!/usr/bin/env python3
r"""Fresh-wheel HTTP/PydanticAI smoke test on remote Linux/Python 3.13 only.

    python scripts/smoke_optional_integrations.py \
        --wheel /candidate/gemmadecision-0.2.0-py3-none-any.whl \
        --output /results/optional-integrations.json --timeout 300

After publication, use --package-version 0.2.0 instead of --wheel. A clean
venv installs only the base package plus serve,pydantic-ai extras. Real local
ONNX/PydanticAI inference runs in a subprocess that exits before a real
Granian server starts, so the two model copies never coexist. Both processes
share a fresh temporary HF cache; no token or mounted model is required.

The overall deadline includes installation and downloading. This is a
functional check, not an accuracy or speed benchmark. No resources are
created on import or --help. Never run the real check on a developer Mac.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse


FORBIDDEN_IMPORTS = {"torch", "torchvision", "torchaudio", "transformers", "vllm", "triton"}
FORBIDDEN_DISTRIBUTIONS = FORBIDDEN_IMPORTS | {"safetensors", "cuda-python"}
STATE = "The same card payment appears twice on my statement."
CHOICES = {"billing": "Investigate duplicate payments and refunds.",
           "technical": "Investigate app crashes and sign-in failures."}
QUESTION = "Which support team should handle this request?"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def require_remote_python():
    if sys.platform != "linux" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Run real inference only on remote Linux with Python 3.13")


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("Optional-integration validation reached its overall deadline")
    return value


def run_group(command, *, env, cwd, log, deadline):
    """Terminate only this invocation's process group, including descendants."""
    with Path(log).open("w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=remaining(deadline))
            if code:
                raise RuntimeError(f"Subprocess exited with status {code}; see {Path(log).name}")
        finally:
            # HTTP worker children inherit this group, including Granian workers.
            # Cleanup also runs after an otherwise successful worker exit.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    process.wait(timeout=min(2, max(0.01, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def install_identity():
    distributions = {
        re.sub(r"[-_.]+", "-", item.metadata["Name"]).lower(): item.version
        for item in importlib.metadata.distributions()
    }
    bad = sorted(name for name in distributions if name in FORBIDDEN_DISTRIBUTIONS
                 or name.startswith(("nvidia-", "cuda-")))
    if bad:
        raise RuntimeError(f"Optional integration install pulled prohibited runtimes: {bad}")
    import gemmadecision
    from gemmadecision import constants
    path = Path(gemmadecision.__file__).resolve()
    if not path.is_relative_to(Path(sys.prefix).resolve()) or "site-packages" not in path.parts:
        raise RuntimeError("GemmaDecision was not imported from the isolated venv installation")
    if not constants.ONNX_REVISION or not constants.ONNX_MANIFEST_SHA256:
        raise RuntimeError("The installed package has no published immutable ONNX pins")
    return {
        "version": importlib.metadata.version("gemmadecision"),
        "package_source": str(path), "python": sys.version.split()[0],
        "distributions": distributions, "forbidden_distributions": bad,
        "model_pins": {"source_revision": constants.MODEL_REVISION,
                       "onnx_revision": constants.ONNX_REVISION,
                       "onnx_manifest_sha256": constants.ONNX_MANIFEST_SHA256},
    }


def verify_backend(metadata, pins):
    assert metadata["backend"] == "onnx" and metadata["device"] == "cpu"
    assert metadata["verified"] is True
    assert metadata["model_revision"] == pins["source_revision"]
    assert metadata["onnx_revision"] == pins["onnx_revision"]
    assert metadata["onnx_manifest_sha256"] == pins["onnx_manifest_sha256"]


def local_check(pins):
    from typing import Literal
    from pydantic_ai import Agent
    from gemmadecision import GemmaDecisionModel
    from gemmadecision.simple import get_engine

    model = GemmaDecisionModel.local()
    agent = Agent(model, output_type=Literal["billing", "technical"], instructions=QUESTION)
    result = agent.run_sync(STATE)
    assert result.output in CHOICES
    backend = get_engine().metadata()
    verify_backend(backend, pins)
    assert model.base_url == "local://gemmadecision"
    return {"typed_literal_output": result.output, "provider": model.base_url,
            "backend": backend, "accuracy_asserted": False}


def http_check(args, deadline, pins):
    import httpx
    from gemmadecision import DecisionClient

    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    key = secrets.token_urlsafe(32)
    environment = dict(os.environ, GEMMADECISION_API_KEY=key, HF_HUB_OFFLINE="1")
    server_log = Path(args.output).with_suffix(".server.log")
    with server_log.open("w") as stream:
        # Inherit the worker process group so the parent can kill all descendants
        # on its deadline. The key is passed only in the private environment.
        process = subprocess.Popen([
            sys.executable, "-I", "-m", "gemmadecision.cli", "serve",
            "--backend", "onnx", "--device", "cpu", "--offline",
            "--host", "127.0.0.1", "--port", str(port), "--request-timeout", "30",
        ], env=environment, stdout=stream, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=base_url, timeout=2) as client:
                while True:
                    remaining(deadline)
                    if process.poll() is not None:
                        raise RuntimeError("Granian exited before readiness; see server log")
                    try:
                        health = client.get("/ready")
                        if health.status_code == 200:
                            metadata = health.json()
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(min(0.2, remaining(deadline)))
                verify_backend(metadata, pins)
                assert metadata["serving_runtime"] == "granian-rust"
                unauthenticated = client.post("/v1/rank", json={"state": STATE, "candidates": CHOICES})
                assert unauthenticated.status_code == 401
            with DecisionClient(base_url, api_key=key, timeout=min(30, remaining(deadline))) as client:
                choice = client.decide(STATE, CHOICES, question=QUESTION)
                assert choice.choice in CHOICES
                assert set(choice.probabilities) == set(CHOICES)
                assert math.isclose(sum(choice.probabilities.values()), 1, abs_tol=1e-6)

            async def native_remote():
                from typing import Literal
                from pydantic_ai import Agent
                from gemmadecision import GemmaDecisionModel
                async with GemmaDecisionModel(base_url=base_url, api_key=key) as model:
                    agent = Agent(model, output_type=Literal["billing", "technical"], instructions=QUESTION)
                    result = await agent.run(STATE, model_settings={"timeout": min(30, remaining(deadline))})
                    assert result.output in CHOICES
                    return result.output

            remote_choice = asyncio.run(native_remote())
            return {"readiness": metadata, "unauthenticated_status": 401,
                    "sdk_choice": choice.model_dump(mode="json"),
                    "native_remote_literal": remote_choice, "server_started_offline": True,
                    "accuracy_asserted": False}
        finally:
            process.terminate()
            try:
                process.wait(timeout=min(3, max(0.01, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def worker(args):
    import importlib.abc

    attempts = []
    class NoTrainingImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in FORBIDDEN_IMPORTS:
                attempts.append(fullname)
                raise RuntimeError(f"Unexpected training/GPU import: {fullname}")
    sys.meta_path.insert(0, NoTrainingImports())
    deadline = time.monotonic() + args.timeout
    report = {"status": "running", "stage": args.worker}
    try:
        report["installation"] = install_identity()
        pins = report["installation"]["model_pins"]
        report["checks"] = local_check(pins) if args.worker == "local" else http_check(args, deadline, pins)
        loaded = sorted(name for name in sys.modules if name.split(".")[0] in FORBIDDEN_IMPORTS)
        assert not attempts and not loaded
        report.update(status="passed", forbidden_import_attempts=attempts, forbidden_loaded_modules=loaded)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      forbidden_import_attempts=attempts)
        raise
    finally:
        write_json(args.output, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--wheel", type=Path)
    source.add_argument("--package-version")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--worker", choices=["local", "http"], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 300:
        parser.error("timeout must be greater than zero and at most 300 seconds")
    require_remote_python()
    if args.worker:
        worker(args)
        return 0
    if (args.wheel is None) == (args.package_version is None):
        parser.error("Choose --wheel or --package-version")
    if args.package_version and not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", args.package_version):
        parser.error("package-version must be an exact version")
    if args.wheel:
        args.wheel = args.wheel.resolve()
        if not args.wheel.is_file() or args.wheel.suffix != ".whl":
            parser.error("wheel must be an existing .whl file")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.timeout
    report = {"status": "running", "timeout_s": args.timeout,
              "scope": "Fresh installed package plus serve,pydantic-ai; real CPU ONNX, local native Agent, Granian HTTP SDK and remote Agent. Functional checks only.",
              "script_sha256": digest(Path(__file__)), "stages": {}}
    if args.wheel:
        report["candidate_wheel"] = {"name": args.wheel.name, "sha256": digest(args.wheel)}
    else:
        report["pypi_version"] = args.package_version
    write_json(args.output, report)
    logs = args.output.with_suffix("")
    logs.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
                "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_OFFLINE",
                "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS", "PIP_NO_INDEX",
                "GEMMADECISION_MODEL_PATH", "GEMMADECISION_CONFIG", "GEMMADECISION_BASE_URL", "GEMMADECISION_API_KEY"):
        environment.pop(key, None)
    environment.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1",
                       HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                       PYDANTIC_AI_NO_BANNER="1", PYTHONUNBUFFERED="1")
    try:
        with tempfile.TemporaryDirectory(prefix="gemmadecision-extras-") as directory:
            root = Path(directory)
            environment["HF_HOME"] = str(root / "hf-cache")
            python = root / "venv" / "bin" / "python"
            script = root / "smoke.py"
            shutil.copy2(__file__, script)
            run_group([sys.executable, "-I", "-m", "venv", str(root / "venv")],
                      env=environment, cwd=root, log=logs / "venv.log", deadline=deadline)
            requirement = (f"gemmadecision[serve,pydantic-ai] @ {args.wheel.as_uri()}" if args.wheel
                           else f"gemmadecision[serve,pydantic-ai]=={args.package_version}")
            pip_report = logs / "pip-report.json"
            run_group([str(python), "-I", "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
                       "--only-binary=:all:", "--timeout", "15", "--retries", "1", "--report", str(pip_report), requirement],
                      env=environment, cwd=root, log=logs / "install.log", deadline=deadline)
            packages = json.loads(pip_report.read_text())["install"]
            own = [item for item in packages if item["metadata"]["name"].lower() == "gemmadecision"]
            assert len(own) == 1
            installed = own[0]
            archive = installed["download_info"]
            wheel_sha = archive.get("archive_info", {}).get("hashes", {}).get("sha256")
            if args.wheel:
                assert wheel_sha == report["candidate_wheel"]["sha256"]
            else:
                assert installed["metadata"]["version"] == args.package_version
                assert urlparse(archive["url"]).hostname == "files.pythonhosted.org"
                assert wheel_sha and re.fullmatch(r"[0-9a-f]{64}", wheel_sha)
            report["installed_artifact"] = {"version": installed["metadata"]["version"],
                                             "url": archive["url"], "sha256": wheel_sha}
            write_json(args.output, report)
            for stage in ("local", "http"):
                stage_result = logs / f"{stage}.json"
                report["active_stage"] = stage
                write_json(args.output, report)
                try:
                    run_group([str(python), "-I", str(script), "--worker", stage, "--output", str(stage_result),
                               "--timeout", str(remaining(deadline))], env=environment, cwd=root,
                              log=logs / f"{stage}.log", deadline=deadline)
                finally:
                    if stage_result.is_file():
                        report["stages"][stage] = json.loads(stage_result.read_text())
                result = report["stages"][stage]
                assert result["status"] == "passed"
                assert result["installation"]["version"] == report["installed_artifact"]["version"]
                write_json(args.output, report)
        report["status"] = "passed"
        report.pop("active_stage", None)
        return 0
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        write_json(args.output, report)
        print(json.dumps({"status": report["status"], "report": str(args.output), "elapsed_s": report["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
