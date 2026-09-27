#!/usr/bin/env python3
"""Measure serving kernels against the frozen v4 reference on one CUDA GPU.

This is a latency/parity experiment with hand-authored unlabeled inputs, not
an accuracy benchmark. It never downloads data, modifies weights, or runs a
generative model. Run on the GPU host, not a laptop.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

REFERENCE_SOURCE_SHA256 = {
    "joint_deployment.py": "2e5212fe7bf6bd40fc9414c8aa480184996eb2416fafc9e2aee57b183c851c64",
    "common.py": "7a41cecc445af13a1b88902920ebdc1cc0e854a4379c243302c4237c29dab307",
    "clm_schema.py": "52cec58afbf49ad7b7aa6bdb7e7476ee42bf3fd7a2703d44319dc4b565987335",
    "clm_heads.py": "3f3b880e940a47b45879614b140fd873f7de9b13ccb8254b07989af7ea92e093",
}


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def softmax(values, temperature):
    top = max(values)
    exponents = [math.exp((value - top) / temperature) for value in values]
    total = math.fsum(exponents)
    return [value / total for value in exponents]


def compare(reference, observed, cases, temperature):
    if len(reference) != len(observed) or not all(math.isfinite(x) for x in observed):
        raise ValueError("Incomplete or non-finite backend scores")
    differences = [abs(a - b) for a, b in zip(reference, observed)]
    agreement, max_probability_error, offset, per_case = 0, 0., 0, []
    for case in cases:
        count = len(case["candidates"])
        baseline, candidate = reference[offset:offset + count], observed[offset:offset + count]
        expected = max(range(count), key=baseline.__getitem__)
        actual = max(range(count), key=candidate.__getitem__)
        agreement += expected == actual
        probs_a, probs_b = softmax(baseline, temperature), softmax(candidate, temperature)
        error = max(abs(a - b) for a, b in zip(probs_a, probs_b))
        max_probability_error = max(max_probability_error, error)
        sorted_reference = sorted(baseline, reverse=True)
        per_case.append({
            "id": case["id"], "reference_top1": expected, "backend_top1": actual,
            "top1_agrees": expected == actual,
            "reference_top1_margin": sorted_reference[0] - sorted_reference[1],
            "max_abs_score_error": max(abs(a - b) for a, b in zip(baseline, candidate)),
            "max_abs_probability_error": error,
        })
        offset += count
    return {
        "max_abs_score_error": max(differences),
        "mean_abs_score_error": statistics.mean(differences),
        "top1_agreement_count": agreement,
        "case_count": len(cases),
        "top1_agreement": agreement / len(cases),
        "max_abs_probability_error": max_probability_error,
        "softmax_temperature": temperature,
        "cases": per_case,
        "interpretation": "Numerical/ranking agreement with frozen inference, not accuracy.",
    }


def make_fixtures(tokenizer, render_state):
    actions = [
        "Check the transaction identifiers and investigate a duplicated card payment.",
        "Explain the invoice breakdown and ask which line item needs clarification.",
        "Help reset the account password after verifying the account ownership.",
        "Review delivery tracking and contact the shipping team about the parcel.",
        "Ask the customer to describe the error displayed during checkout.",
        "Escalate the suspected unauthorized transaction to the fraud support team.",
        "Confirm the subscription renewal date and explain the cancellation policy.",
        "Collect the order number and check whether the requested refund was processed.",
        "Check the service status page for a known technical outage.",
        "Update the mailing address after confirming the customer's request.",
        "Explain available plans without changing the customer's current subscription.",
        "Investigate the missing receipt and send the existing payment confirmation.",
        "Ask for more information about the problem before choosing a support action.",
        "Review notification settings to understand why the customer missed an alert.",
        "Check whether the merchant placed a temporary authorization hold on the card.",
        "Summarize the available evidence and transfer the case to a human support agent.",
    ]
    question = "Which support action best fits the current request and the available information?"
    short_states = [
        "The customer sees two identical card charges after placing one order yesterday.",
        "A customer cannot sign in and says the password reset email has not arrived.",
        "The invoice total changed this month, and the customer wants an explanation.",
        "The tracking page says delivered, but the customer cannot find the package.",
        "The customer reports an unfamiliar payment and also asks where to find their receipt.",
    ]
    def count(text):
        return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    history = []
    for index in range(1, 100):
        history.append(
            f"Handover record {index}: the agent checked the existing case notes, "
            "recorded the customer's preferred email contact, and left the account settings unchanged."
        )
        if count("\n".join(history)) >= 650:
            break
    long_prefix = "Background support handover log:\n" + "\n".join(history)
    cases = []
    for group in ("short", "long"):
        for index, state in enumerate(short_states):
            count_candidates = 16 if index == 4 else 4
            selected = actions if count_candidates == 16 else actions[index * 3:index * 3 + 4]
            state = f"Case {group}-{index}. {state}"
            if group == "long":
                state = long_prefix + "\n\nCurrent request: " + state
            cases.append({"id": f"{group}-{index}", "group": group, "state": state,
                          "question": question, "candidates": selected})
    for index in range(2):
        cases.append({
            "id": f"structured-{index}", "group": "structured", "question": question,
            "state": {"case": f"structured-{index}", "request": short_states[index],
                      "recent_events": history[:5], "customer_contacted": True},
            "candidates": actions,
        })
    for case in cases:
        rendered = render_state(case["state"], case["question"])
        case["rendered_state"] = rendered
        case["state_tokens"] = count(rendered)
        case["texts"] = [rendered + "\n\nCandidate action:\n" + candidate
                         for candidate in case["candidates"]]
        case["pair_tokens"] = [count(text) for text in case["texts"]]
        if not 1 <= case["state_tokens"] <= 2048:
            raise ValueError("Fixture state exceeds the frozen model's state limit")
        if case["group"] == "long" and case["state_tokens"] <= 512:
            raise ValueError("Long fixture did not exercise the sliding attention boundary")
        if any(count(candidate) > 768 for candidate in case["candidates"]):
            raise ValueError("Fixture candidate exceeds the frozen candidate limit")
    if sum(len(case["texts"]) for case in cases) != 96:
        raise AssertionError("The fixed fixture must contain 96 pairs")
    return cases


class FrozenReference:
    def __init__(self, model_path):
        import torch
        for name, expected in REFERENCE_SOURCE_SHA256.items():
            if hashlib.sha256((model_path / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen reference source changed: {name}")
        for name in ("common", "clm_schema"):
            if name in sys.modules:
                module_path = Path(sys.modules[name].__file__).resolve()
                if module_path != model_path / f"{name}.py":
                    raise RuntimeError(f"Conflicting reference dependency already imported: {name}")
        sys.path.insert(0, str(model_path))
        spec = importlib.util.spec_from_file_location("frozen_v4_reference", model_path / "joint_deployment.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.GemmaJointRanker(model_path, device="cuda")
        self._torch = torch
        self.metadata = {"backend": "published_v4_singleton", "device": "cuda",
                         "dtype": "bfloat16", "head_dtype": "float32",
                         "source_sha256": REFERENCE_SOURCE_SHA256}

    def score_pairs(self, texts):
        with self._torch.inference_mode():
            return [self.model._score(self.model._tokens(text)) for text in texts]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path,
                        help="Complete immutable v4 bundle including reference .py files")
    parser.add_argument("--backend", choices=("torch", "vllm"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-seconds", type=int, default=420,
                        help="Stop between measured calls after this elapsed duration")
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 3 or args.max_seconds < 1:
        parser.error("Use at least one warmup, three repeats, and a positive time bound")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite a previous measurement: {args.output}")
    args.model_path = args.model_path.expanduser().resolve()
    started = time.perf_counter()
    report = {"format": "gemmadecision-serving-benchmark-v1", "status": "initializing",
              "requested_backend": args.backend, "stages": {},
              "accuracy_measured": False, "cache_enabled": False,
              "scope": "Local score_pairs including tokenization/head and GPU synchronization; excludes HTTP, model download, and request validation.",
              "notes": ["Hand-authored unlabeled inputs; no JevBench examples or answer keys.",
                        "No training, calibration fitting, quantization, or model changes.",
                        "p95 is an interpolated percentile of only five samples by default, not a production tail-latency estimate.",
                        "Cold load is this process's initialization, not uncached network download.",
                        "vLLM uses a CPU FP32 head; its worker GPU memory is not represented by parent-process torch memory metrics."]}

    def checkpoint():
        report["elapsed_s"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    def check_deadline():
        if time.perf_counter() - started > args.max_seconds:
            raise TimeoutError("Benchmark elapsed-time limit reached between inference calls")

    try:
        import torch
        from transformers import AutoTokenizer
        from gemmadecision.backends.torch import TorchBackend, verify_model_files, MODEL_REVISION
        from gemmadecision.engine import render_state
        from gemmadecision.constants import TEMPERATURE
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This experiment requires a CUDA GPU with BF16 support")
        torch.manual_seed(0)
        verify_model_files(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                                  trust_remote_code=False)
        cases = make_fixtures(tokenizer, render_state)
        texts = [text for case in cases for text in case["texts"]]
        report["model_revision"] = MODEL_REVISION
        report["fixture_sha256"] = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
        report["fixtures"] = cases
        props = torch.cuda.get_device_properties(0)
        versions = {}
        for dependency in ("torch", "transformers", "vllm", "safetensors", "tokenizers", "gemmadecision"):
            try:
                versions[dependency] = importlib.metadata.version(dependency)
            except importlib.metadata.PackageNotFoundError:
                versions[dependency] = None
        report["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                                 "packages": versions, "cuda_runtime": torch.version.cuda,
                                 "gpu_name": props.name, "gpu_memory_bytes": props.total_memory,
                                 "compute_capability": [props.major, props.minor],
                                 "torch_threads": torch.get_num_threads(),
                                 "torch_tf32_matmul": torch.backends.cuda.matmul.allow_tf32}
        report["warmups_per_workload"] = args.warmups
        report["repeats_per_workload"] = args.repeats
        worksets = []
        for group in ("short", "long"):
            pool = [text for case in cases if case["group"] == group for text in case["texts"]]
            token_pool = [n for case in cases if case["group"] == group for n in case["pair_tokens"]]
            for size in (1, 4, 16, 32):
                worksets.append({"name": f"{group}-{size}-pairs", "texts": pool[:size],
                                 "tokens": token_pool[:size], "group": group, "pairs": size})
        report["status"] = "running"
        checkpoint()

        def measure(name, factory):
            check_deadline()
            print(f"Serving benchmark: loading {name}", flush=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            stage_started = time.perf_counter()
            before = time.perf_counter()
            backend = factory()
            torch.cuda.synchronize()
            stage = {"load_s": time.perf_counter() - before, "workloads": {}}
            metadata = getattr(backend, "metadata", {})
            stage["metadata"] = metadata() if callable(metadata) else metadata
            report["stages"][name] = stage
            checkpoint()
            try:
                # Match rendering against the actual frozen source before comparisons.
                if isinstance(backend, FrozenReference):
                    import common
                    for case in cases:
                        if common.schema_state(case["state"], case["question"]) != case["rendered_state"]:
                            raise AssertionError("Package and frozen reference state rendering differ")
                before = time.perf_counter()
                first = backend.score_pairs(worksets[0]["texts"])
                torch.cuda.synchronize()
                stage["first_request_s"] = time.perf_counter() - before
                if not all(math.isfinite(x) for x in first):
                    raise ValueError("Non-finite first-request scores")
                stage["scores"] = backend.score_pairs(texts)
                torch.cuda.synchronize()
                if len(stage["scores"]) != len(texts) or not all(math.isfinite(x) for x in stage["scores"]):
                    raise ValueError("Invalid scores in the parity pass")
                if name != "frozen_reference":
                    stage["reference_parity"] = compare(
                        report["stages"]["frozen_reference"]["scores"], stage["scores"], cases, TEMPERATURE)
                checkpoint()
                for workset in worksets:
                    check_deadline()
                    for _ in range(args.warmups):
                        backend.score_pairs(workset["texts"])
                    torch.cuda.synchronize()
                    timings = []
                    measured_scores = []
                    for _ in range(args.repeats):
                        check_deadline()
                        torch.cuda.synchronize()
                        before = time.perf_counter()
                        scores = backend.score_pairs(workset["texts"])
                        torch.cuda.synchronize()
                        timings.append(time.perf_counter() - before)
                        measured_scores.append(scores)
                    first_scores = measured_scores[0]
                    repeat_variation = max(abs(a - b) for row in measured_scores
                                           for a, b in zip(first_scores, row))
                    median = statistics.median(timings)
                    measurement = {"input_pairs": workset["pairs"], "input_token_counts": workset["tokens"],
                                   "samples_s": timings, "p50_ms": median * 1000,
                                   "p95_ms": percentile(timings, .95) * 1000,
                                   "pairs_per_second": workset["pairs"] / median,
                                   "input_tokens_per_second": sum(workset["tokens"]) / median,
                                   "max_repeat_score_variation": repeat_variation,
                                   "scores": first_scores}
                    if name != "frozen_reference":
                        base = report["stages"]["frozen_reference"]["workloads"][workset["name"]]
                        measurement["speedup_vs_frozen_reference"] = base["p50_ms"] / measurement["p50_ms"]
                        measurement["max_abs_score_error_vs_same_workload"] = max(
                            abs(a - b) for a, b in zip(base["scores"], first_scores))
                    stage["workloads"][workset["name"]] = measurement
                    print(f"{name} {workset['name']}: p50={median * 1000:.2f}ms; "
                          f"{workset['pairs'] / median:.1f} pairs/s", flush=True)
                    checkpoint()
            finally:
                stage["elapsed_s"] = time.perf_counter() - stage_started
                stage["peak_gpu_allocated_bytes"] = (
                    None if name == "vllm" else torch.cuda.max_memory_allocated())
                stage["peak_gpu_reserved_bytes"] = (
                    None if name == "vllm" else torch.cuda.max_memory_reserved())
                stage["gpu_memory_metric_scope"] = (
                    "Unavailable: vLLM worker process owns decoder memory" if name == "vllm"
                    else "torch.cuda allocator in this process, excluding driver/non-Torch allocations")
                del backend
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        measure("frozen_reference", lambda: FrozenReference(args.model_path))
        measure("torch_strict", lambda: TorchBackend(args.model_path, device="cuda", strict=True))
        measure("torch_batched", lambda: TorchBackend(args.model_path, device="cuda", max_batch_size=32))
        if args.backend == "vllm":
            from gemmadecision.backends.vllm import VLLMBackend
            measure("vllm", lambda: VLLMBackend(args.model_path,
                    export_path=args.output.parent / "vllm-encoder", max_num_seqs=32))
        report["status"] = "complete"
        report["interpretation"] = "Measured parity and latency on fixed unlabeled fixtures; not model quality, JevBench, or an HTTP load test."
        checkpoint()
        print(f"Complete: {args.output}", flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
        checkpoint()
        raise


if __name__ == "__main__":
    main()
