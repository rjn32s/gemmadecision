#!/usr/bin/env python3
"""Bounded development-only parity and CPU timing of a frozen ONNX export.

Run on the export host, never a laptop. The fixed development file is verified
before parsing; calibration, final, and public benchmark files are not opened.
No model training, thresholds/temperature fitting, publication, or network.
Reports contain aggregate metrics, hashes, and synthetic workload lengths, not
dataset text, gold labels, or individual candidate predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import tempfile
import time


DEVELOPMENT_SHA256 = "26c7628d09cd1f44d5a77a10130f9e2ede63abc88b286c33c6ad6a9b98436a51"
MODEL_REVISION = "785d530221c990671f29976902540101bb9c7647"
PROVISIONAL_MINIMUM_AGREEMENT = 0.99


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(value):
    return hashlib.sha256(value.encode()).hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class Deadline:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.ends = self.started + seconds

    def check(self):
        if time.monotonic() >= self.ends:
            raise TimeoutError("Validation process deadline reached")

    def elapsed(self):
        return time.monotonic() - self.started


def load_development(path, maximum):
    if digest(path) != DEVELOPMENT_SHA256:
        raise ValueError("Development data does not match the frozen v4 development SHA256")
    rows = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != "development":
                raise ValueError("Only rows explicitly marked development may be inspected")
            if not isinstance(row.get("id"), str) or row.get("family") not in {
                "intent_routing", "evidence_relation", "rule_compliance"
            }:
                raise ValueError("Unexpected development row schema")
            candidates, target = row.get("candidates"), row.get("target")
            if (not isinstance(candidates, list) or not 2 <= len(candidates) <= 64
                    or any(not isinstance(item, str) for item in candidates)
                    or not isinstance(target, int) or not 0 <= target < len(candidates)):
                raise ValueError("Invalid development candidate structure")
            rows.append(row)
    if len(rows) != 700 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Frozen development file must contain 700 distinct rows")
    # Selection is independent of every model's predictions and labels.
    rows.sort(key=lambda row: identity("onnx-release-parity-v1:" + row["id"]))
    return rows[:maximum], len(rows)


def prepare_development(rows, model_path):
    from tokenizers import Tokenizer
    from gemmadecision.engine import DecisionEngine
    from gemmadecision.types import RankRequest

    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()

    class CountOnlyBackend:
        max_state_tokens = 2048
        max_action_tokens = 768
        context_limit = 3072
        max_batch_tokens = 4096

        @staticmethod
        def count_tokens(text):
            return len(tokenizer.encode(text, add_special_tokens=True).ids)

    engine = DecisionEngine(CountOnlyBackend(), cache_size=0)
    prepared, skipped = [], []
    for row in rows:
        try:
            plan = engine.prepare_rank(RankRequest(
                state=row["state"], question=row.get("question", ""), candidates=row["candidates"]))
        except ValueError as error:
            # Preserve the fixed length/refusal contract. Never truncate.
            message = str(error)
            if "limit" not in message and "exceeds" not in message:
                raise ValueError("Invalid development request schema") from error
            skipped.append({"id_sha256": identity(row["id"]), "family": row["family"],
                            "reason": "published_token_limit"})
            continue
        prepared.append({"id_sha256": identity(row["id"]), "family": row["family"],
                         "target": row["target"], "texts": plan.texts, "tokens": plan.tokens})
    return prepared, skipped


def score_development(backend, cases, deadline, checkpoint):
    scores = []
    chunk, size = [], 0
    chunks = []
    for index, case in enumerate(cases):
        if chunk and size + len(case["texts"]) > 128:
            chunks.append(chunk)
            chunk, size = [], 0
        chunk.append(index)
        size += len(case["texts"])
    if chunk:
        chunks.append(chunk)
    started = time.perf_counter()
    for indices in chunks:
        deadline.check()
        texts = [text for index in indices for text in cases[index]["texts"]]
        values = backend.score_pairs(texts)
        if len(values) != len(texts) or not all(math.isfinite(value) for value in values):
            raise RuntimeError("Backend returned invalid development scores")
        offset = 0
        for index in indices:
            count = len(cases[index]["texts"])
            scores.append(values[offset:offset + count])
            offset += count
        checkpoint(len(scores), len(cases))
    return scores, time.perf_counter() - started


def accuracy_summary(cases, scores):
    summary = defaultdict(lambda: {"cases": 0, "correct": 0})
    for case, values in zip(cases, scores, strict=True):
        correct = max(range(len(values)), key=values.__getitem__) == case["target"]
        for name in ("overall", case["family"]):
            summary[name]["cases"] += 1
            summary[name]["correct"] += correct
    for value in summary.values():
        value["accuracy"] = value["correct"] / value["cases"]
    return dict(summary)


def compare(cases, reference, candidate):
    from gemmadecision.engine import probabilities

    summary = defaultdict(lambda: {"cases": 0, "top1_agreements": 0, "max_abs_score_error": 0.0,
                                   "sum_abs_score_error": 0.0, "pairs": 0})
    margins, disagreements, errors, probability_errors = [], [], [], []
    for case, expected, actual in zip(cases, reference, candidate, strict=True):
        if len(expected) != len(actual):
            raise ValueError("Mismatched candidate population")
        left = max(range(len(expected)), key=expected.__getitem__)
        right = max(range(len(actual)), key=actual.__getitem__)
        ordered = sorted(expected, reverse=True)
        margin = ordered[0] - ordered[1]
        error = [abs(a - b) for a, b in zip(expected, actual, strict=True)]
        probability_error = max(abs(a - b) for a, b in zip(probabilities(expected), probabilities(actual), strict=True))
        margins.append(margin)
        errors.extend(error)
        probability_errors.append(probability_error)
        band = "margin_lt_0.1" if margin < .1 else "margin_lt_0.5" if margin < .5 else "margin_lt_1" if margin < 1 else "margin_ge_1"
        for key in ("overall", case["family"], band):
            item = summary[key]
            item["cases"] += 1
            item["top1_agreements"] += left == right
            item["max_abs_score_error"] = max(item["max_abs_score_error"], max(error))
            item["sum_abs_score_error"] += sum(error)
            item["pairs"] += len(error)
        if left != right:
            disagreements.append({"id_sha256": case["id_sha256"], "family": case["family"],
                                  "reference_margin": margin, "max_abs_score_error": max(error),
                                  "max_abs_probability_error": probability_error})
    for item in summary.values():
        item["top1_agreement"] = item["top1_agreements"] / item["cases"]
        item["mean_abs_score_error"] = item.pop("sum_abs_score_error") / item["pairs"]
    return {"groups": dict(summary), "disagreements": disagreements,
            "max_abs_score_error": max(errors), "mean_abs_score_error": statistics.mean(errors),
            "p95_abs_score_error": percentile(errors, .95),
            "max_abs_probability_error": max(probability_errors),
            "reference_margin_p05": percentile(margins, .05),
            "reference_margin_median": statistics.median(margins),
            "minimum_top1_agreement": PROVISIONAL_MINIMUM_AGREEMENT,
            "provisional_agreement_gate_passed": summary["overall"]["top1_agreement"] >= PROVISIONAL_MINIMUM_AGREEMENT,
            "interpretation": "Conversion fidelity on existing development data, not a new blind accuracy benchmark."}


@contextmanager
def model_view(export_path, manifest_name, output, expected_digest=None):
    """Give the real backend its normal manifest name without editing exports."""
    manifest_path = export_path / manifest_name
    manifest_digest = digest(manifest_path)
    if expected_digest is not None and manifest_digest != expected_digest:
        raise ValueError("Supplied trusted manifest digest does not match the export")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_model_revision") != MODEL_REVISION:
        raise ValueError("Export manifest refers to another model")
    with tempfile.TemporaryDirectory(prefix="validation-model-view-", dir=output) as directory:
        view = Path(directory)
        for name in manifest["files"]:
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Export files must have flat relative names")
            (view / name).symlink_to((export_path / name).resolve())
        (view / "onnx_manifest.json").write_bytes(manifest_path.read_bytes())
        yield view, manifest_digest


def runtime_smoke(backend):
    from gemmadecision.engine import DecisionEngine
    from gemmadecision.types import Choice, Noul, Score, ChoiceAnswer, NoulAnswer, ScoreAnswer

    engine = DecisionEngine(backend, cache_size=0)
    response = engine.system_one(
        "The customer reports two charges for one order.",
        {"team": Choice(instructions="Choose the support team.", criteria={"billing": "Billing support", "technical": "Technical support"}),
         "payment": Noul(instructions="Is the request about a payment?"),
         "priority": Score(instructions="Choose the urgency level.", criteria=["Routine request", "Urgent request"])},
    )
    assert isinstance(response.answers["team"], ChoiceAnswer)
    assert isinstance(response.answers["payment"], NoulAnswer)
    assert isinstance(response.answers["priority"], ScoreAnswer)
    assert response.usage.cached_pairs == 0
    assert all(math.isfinite(value) for answer in response.answers.values() for value in answer.scores.values())
    assert backend.score_pairs([]) == []
    # Assert real runtime reorders varied length batches correctly.
    texts = ["Support request.\n\nCandidate action:\nCheck billing.",
             "The customer has a long support history. " * 45 + "\n\nCandidate action:\nInvestigate payment.",
             "Password reset.\n\nCandidate action:\nRestore account access."]
    original = backend.score_pairs(texts)
    reversed_scores = backend.score_pairs(list(reversed(texts)))
    reorder_error = max(abs(a - b) for a, b in zip(original, reversed(reversed_scores), strict=True))
    if reorder_error > 1e-5:
        raise RuntimeError("Backend did not preserve original request order")
    return {"choice_noul_score_typed_output": True, "cache_disabled": True,
            "empty_batch": True, "batch_reversal_max_abs_error": reorder_error,
            "accuracy_asserted": False}


def latency_workloads(backend, deadline, progress):
    from gemmadecision.engine import DecisionEngine, render_state

    engine = DecisionEngine(backend, cache_size=0)
    question = "Choose the support team."
    choices = ["billing", "technical", "account", "other"]
    base = "I was charged twice."
    sentence = " The support agent recorded the customer's account details."
    filler = " Note."

    def count(state, candidate):
        return backend.count_tokens(render_state(state, question) + "\n\nCandidate action:\n" + candidate)

    results = []
    for target in (30, 128, 512):
        state = base
        while count(state + sentence, choices[0]) <= target:
            state += sentence
        while count(state + filler, choices[0]) <= target:
            state += filler
        for candidate_count in (2, 4):
            selected = choices[:candidate_count]
            repeats = 5 if target == 512 else 10
            actual_tokens = [count(state, choice) for choice in selected]
            for _ in range(3):
                deadline.check()
                result = engine.rank(state, selected, question=question)
                if result.usage.cached_pairs != 0:
                    raise RuntimeError("Latency measurement unexpectedly used a score cache")
            samples = []
            for _ in range(repeats):
                deadline.check()
                started = time.perf_counter()
                result = engine.rank(state, selected, question=question)
                samples.append(time.perf_counter() - started)
                if len(result.ranked) != candidate_count or result.usage.cached_pairs != 0:
                    raise RuntimeError("Invalid timed ranking response")
            measured = {
                "name": f"tokens-{target}-choices-{candidate_count}",
                "target_joint_tokens": target, "actual_joint_token_counts": actual_tokens,
                "candidate_count": candidate_count, "warmups": 3, "repeats": repeats,
                "samples_s": samples, "p50_ms": percentile(samples, .5) * 1000,
                "p95_ms": percentile(samples, .95) * 1000,
                "sequential_requests_per_s": repeats / sum(samples),
                "scope": "DecisionEngine.rank including validation, tokenization, scoring and typed response; no HTTP/download; cache disabled.",
            }
            results.append(measured)
            progress(measured)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--development-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=700)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-seconds", type=int, default=840)
    parser.add_argument("--primary-manifest-sha256", "--fp32-manifest-sha256", dest="primary_manifest_sha256")
    parser.add_argument("--int8-manifest-sha256")
    parser.add_argument("--skip-int8", action="store_true", help="Compare only the supplied primary manifest against Torch; omit sibling onnx_manifest.int8.json")
    args = parser.parse_args(argv)
    if not 100 <= args.max_cases <= 700 or not 1 <= args.threads <= 64 or not 60 <= args.max_seconds <= 900:
        parser.error("Use 100–700 cases, 1–64 threads, and a 60–900 second deadline")
    deadline = Deadline(args.max_seconds)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "validation-report.json"
    report = {"status": "running", "model_revision": MODEL_REVISION,
              "development_sha256": DEVELOPMENT_SHA256, "development_only": True,
              "calibration_or_final_opened": False, "model_or_temperature_changed": False,
              "validator_sha256": digest(Path(__file__)), "max_seconds": args.max_seconds,
              "requested_cases": args.max_cases, "cpu_threads": args.threads,
              "provisional_top1_agreement_gate": PROVISIONAL_MINIMUM_AGREEMENT,
              "backends": {},
              "latency_caveat": "Small repeated same-process samples, not production tail latency or an accuracy benchmark."}

    def save():
        report["elapsed_seconds"] = deadline.elapsed()
        report["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if platform.system() != "Darwin" else 1024 * 1024)
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(report_path)

    def alarm_handler(signum, frame):
        raise TimeoutError("Validation process deadline reached")

    previous_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(args.max_seconds)
    save()
    try:
        rows, available = load_development(args.development_path, args.max_cases)
        import torch
        from gemmadecision.backends.torch import TorchBackend
        from gemmadecision.backends.onnx import ONNXBackend

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        cpu_models = [line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")] if Path("/proc/cpuinfo").exists() else []
        report["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                                 "cpu_model": cpu_models[0] if cpu_models else platform.processor(),
                                 "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                                 "versions": {name: importlib.metadata.version(name) for name in
                                              ("torch", "transformers", "onnxruntime", "tokenizers", "numpy", "pydantic")}}
        cases, skipped = prepare_development(rows, args.model_path)
        report["selection"] = {"available": available, "selected": len(rows), "supported": len(cases),
                               "skipped": skipped, "selected_ids_sha256": identity("\n".join(identity(row["id"]) for row in rows)),
                               "families": dict(Counter(case["family"] for case in cases)),
                               "selection_rule": "SHA256(onnx-release-parity-v1:<row id>) ascending; independent of labels/predictions.",
                               "pair_count": sum(len(case["texts"]) for case in cases),
                               "joint_token_min": min(min(case["tokens"]) for case in cases),
                               "joint_token_max": max(max(case["tokens"]) for case in cases)}
        del rows
        if len(cases) < 100:
            raise ValueError("At least 100 supported development cases are required")
        save()

        def evaluate(name, load):
            deadline.check()
            started = time.perf_counter()
            backend = load()
            item = {"load_and_integrity_check_s": time.perf_counter() - started,
                    "metadata": backend.metadata, "development_complete": False, "latency": []}
            report["backends"][name] = item
            save()
            try:
                item["runtime_smoke"] = runtime_smoke(backend)

                def checkpoint(done, total):
                    item["development_cases_scored"] = done
                    save()
                    print(f"{name}: development {done}/{total}; elapsed={deadline.elapsed():.1f}s", flush=True)

                values, seconds = score_development(backend, cases, deadline, checkpoint)
                item["development_seconds"] = seconds
                item["development_complete"] = True
                item["development_accuracy"] = accuracy_summary(cases, values)

                def measurement(value):
                    item["latency"].append(value)
                    save()
                    print(f"{name}: {value['name']} p50={value['p50_ms']:.2f}ms", flush=True)

                latency_workloads(backend, deadline, measurement)
                return values
            finally:
                del backend
                gc.collect()
                save()

        reference = evaluate("torch_fp32", lambda: TorchBackend(
            args.model_path, device="cpu", max_batch_tokens=4096, max_batch_size=16))
        # Each temporary view has one backend session at a time; no graph copies
        # and no memory overlap with PyTorch model weights are required.
        with model_view(args.onnx_path, "onnx_manifest.json", args.output, args.primary_manifest_sha256) as (view, manifest_sha):
            values = evaluate("onnx_primary", lambda: ONNXBackend(
                view, num_threads=args.threads, max_batch_tokens=4096, max_batch_size=16,
                manifest_sha256=manifest_sha))
            report["backends"]["onnx_primary"]["manifest_sha256"] = manifest_sha
            report["backends"]["onnx_primary"]["parity"] = compare(cases, reference, values)
            primary_values = values
            save()
        if not args.skip_int8:
            with model_view(args.onnx_path, "onnx_manifest.int8.json", args.output, args.int8_manifest_sha256) as (view, manifest_sha):
                values = evaluate("onnx_int8", lambda: ONNXBackend(
                    view, num_threads=args.threads, max_batch_tokens=4096, max_batch_size=16,
                    manifest_sha256=manifest_sha))
                item = report["backends"]["onnx_int8"]
                item["manifest_sha256"] = manifest_sha
                item["parity"] = compare(cases, reference, values)
                item["parity_to_onnx_primary"] = compare(cases, primary_values, values)
                save()
        report["status"] = "complete"
        report["provisional_primary_gate_passed"] = report["backends"]["onnx_primary"]["parity"]["provisional_agreement_gate_passed"]
        report["provisional_int8_gate_passed"] = (
            report["backends"].get("onnx_int8", {}).get("parity", {}).get("provisional_agreement_gate_passed", False))
        # A failed conversion-quality gate is a complete measurement, not an
        # infrastructure failure. Release tooling must inspect this boolean.
        report["release_selection"] = "No artifact promoted or published by this script."
        save()
        print(json.dumps({"status": report["status"], "cases": len(cases),
                          "provisional_primary_gate_passed": report["provisional_primary_gate_passed"],
                          "provisional_int8_gate_passed": report["provisional_int8_gate_passed"],
                          "elapsed_seconds": deadline.elapsed()}), flush=True)
        return 0
    except Exception as error:
        report["status"] = "partial" if isinstance(error, TimeoutError) else "failed"
        # Keep data-dependent error strings out of shareable measurement files.
        report["error_type"] = type(error).__name__
        save()
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
