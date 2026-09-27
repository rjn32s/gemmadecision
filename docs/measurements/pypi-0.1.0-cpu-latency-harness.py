
import hashlib
from importlib.metadata import distribution, version
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

started = time.monotonic()
report_path = Path(sys.argv[1])
report = {
    "status": "running", "accuracy_evaluated": False, "source_code_mounted": False,
    "device": "cpu", "requested_cpu_cores": 4, "requested_memory_mib": 8192,
    "cache_size": 0, "warmups_per_workload": 3, "measurements": [],
    "scope": "Sequential in-process top-level rank calls, including validation, tokenization, encoder, scalar head and response construction; no HTTP or network in measured warm calls.",
    "percentile_note": "Interpolated p50/p95 from ten samples (five for optional long input); not production tail latency or an accuracy benchmark.",
}

def save():
    report_path.write_text(json.dumps(report, indent=2) + "\n")

def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)

save()
try:
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available()
    import gemmadecision
    from gemmadecision import rank
    from gemmadecision.constants import MODEL_REVISION
    from gemmadecision.model import download_model
    from gemmadecision.simple import get_engine
    from gemmadecision.engine import render_state
    dist = distribution("gemmadecision")
    module = Path(gemmadecision.__file__).resolve()
    assert dist.version == "0.1.0"
    assert MODEL_REVISION == "785d530221c990671f29976902540101bb9c7647"
    assert "site-packages" in module.parts
    assert module.is_relative_to(Path(dist.locate_file("")).resolve())
    assert dist.read_text("direct_url.json") is None
    pip_report = json.loads(Path("/tmp/gemmadecision-install.json").read_text())
    installed = next(row for row in pip_report["install"] if row["metadata"]["name"].lower() == "gemmadecision")
    assert installed["download_info"]["url"].startswith("https://files.pythonhosted.org/")
    model_lines = [line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name")]
    report["installation"] = {
        "package_version": dist.version,
        "distribution_source": "PyPI wheel, installed by pip; no editable/local source",
        "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "pypi_artifact": installed["download_info"],
    }
    report["environment"] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_model": model_lines[0] if model_lines else platform.processor(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "versions": {name: version(name) for name in ("torch", "transformers", "safetensors", "tokenizers", "pydantic")},
    }
    report["imports_and_distribution_checks_s"] = time.monotonic() - started
    download_started = time.perf_counter()
    model_path = download_model()
    report["download_s"] = time.perf_counter() - download_started
    report["public_model"] = {
        "repo": "rajan2k/GemmaDecision-270M", "revision": MODEL_REVISION,
        "authentication": "Anonymous: package downloader uses token=False",
        "cache": "Fresh temporary HF cache; no model in image or mounted volume",
    }
    os.environ["GEMMADECISION_MODEL_PATH"] = str(model_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    load_started = time.perf_counter()
    engine = get_engine()
    report["load_and_integrity_checks_s"] = time.perf_counter() - load_started
    assert engine.backend.device.type == "cpu"
    assert engine.cache_size == 0
    report["engine"] = engine.metadata()
    save()

    question = "Choose the best support team."
    labels = ["billing", "technical", "account", "other"]
    # Fixed synthetic fixture: short customer statement plus repeated neutral
    # context. Token targets describe joint inputs; actual counts are recorded.
    base_state = "The customer was charged twice for one train ticket."
    sentence = " The support agent reviewed the customer's account notes and transaction details."
    filler = " Details noted."

    def joint_tokens(state, candidate):
        return engine.backend.count_tokens(render_state(state, question) + "\n\nCandidate action:\n" + candidate)

    def fixture(target):
        state = base_state
        while joint_tokens(state + sentence, labels[0]) <= target:
            state += sentence
        while joint_tokens(state + filler, labels[0]) <= target:
            state += filler
        return state

    first_state = fixture(32)
    first_started = time.perf_counter()
    first = rank(first_state, choices=labels[:2], question=question)
    report["first_rank_after_load_s"] = time.perf_counter() - first_started
    assert first.usage.cached_pairs == 0
    report["cold_download_load_first_rank_s"] = report["download_s"] + report["load_and_integrity_checks_s"] + report["first_rank_after_load_s"]

    def measure(name, target, candidate_count, repeats):
        state = fixture(target)
        choices = labels[:candidate_count]
        token_counts = [joint_tokens(state, candidate) for candidate in choices]
        for _ in range(3):
            warm = rank(state, choices=choices, question=question)
            assert warm.usage.cached_pairs == 0
        samples = []
        for _ in range(repeats):
            before = time.perf_counter()
            result = rank(state, choices=choices, question=question)
            samples.append(time.perf_counter() - before)
            assert len(result.ranked) == candidate_count
            assert result.usage.cached_pairs == 0
            assert {row.text for row in result.ranked} == set(choices)
        result = {
            "name": name, "candidate_count": candidate_count, "target_joint_tokens": target,
            "actual_joint_token_counts": token_counts,
            "state_tokens": engine.backend.count_tokens(render_state(state, question)),
            "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
            "warmups": 3, "repeats": repeats, "samples_s": samples,
            "p50_ms": percentile(samples, .5) * 1000,
            "p95_ms": percentile(samples, .95) * 1000,
            "min_ms": min(samples) * 1000, "max_ms": max(samples) * 1000,
            "sequential_requests_per_s": repeats / sum(samples),
            "candidate_pairs_per_s": repeats * candidate_count / sum(samples),
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        report["measurements"].append(result)
        save()
        print(f"CPU {name}: p50={result['p50_ms']:.2f}ms p95={result['p95_ms']:.2f}ms; tokens={token_counts}", flush=True)

    for name, target, count in [("short-2", 32, 2), ("short-4", 32, 4), ("medium-2", 128, 2), ("medium-4", 128, 4)]:
        if time.monotonic() - started > 140:
            raise TimeoutError("Insufficient time remains for the required CPU timing workloads")
        measure(name, target, count, 10)
    if time.monotonic() - started < 125:
        measure("long-2", 512, 2, 5)
    else:
        report["optional_long_workload"] = "Skipped to preserve the 180-second process deadline"
    report["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report["status"] = "passed"
except Exception as error:
    report["status"] = "failed"
    report["error"] = f"{type(error).__name__}: {error}"
    report["traceback"] = traceback.format_exc()
finally:
    report["measurement_process_seconds"] = time.monotonic() - started
    save()
sys.exit(0 if report["status"] == "passed" else 1)
