"""Validate a candidate wheel on remote Linux/Python 3.13, with no source checkout.

Each invocation creates a clean venv and fresh Hugging Face cache. The minimal
mode installs only the wheel's ordinary dependencies. The colab mode first
installs real CPU Torch 2.11.0 / torchvision 0.26.0, then verifies that installing
GemmaDecision neither replaces those packages nor imports them during scoring.
This simulates the relevant notebook dependency condition, not all of Colab.

    python scripts/smoke_lightweight_release.py \
        --wheel /candidate/gemmadecision-0.2.0-py3-none-any.whl \
        --mode minimal --output /results/minimal.json

Run colab as a separate bounded CPU function. This script performs real remote
inference and must not be run on a Mac. --help and syntax checks are harmless.
After publication, replace --wheel with --package-version 0.2.0 to verify the
ordinary public PyPI installation and require its artifact URL to be hosted
on files.pythonhosted.org.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


FORBIDDEN_DISTRIBUTIONS = {
    "torch", "torchvision", "torchaudio", "transformers", "safetensors",
    "pydantic-ai", "pydantic-ai-slim", "pydantic-graph", "granian", "fastapi", "starlette",
    "uvicorn", "vllm", "triton", "cuda-python", "cuda-bindings", "cuda-pathfinder",
}
FORBIDDEN_IMPORTS = {
    "torch", "torchvision", "torchaudio", "transformers", "safetensors",
    "pydantic_ai", "granian", "fastapi", "starlette", "uvicorn", "vllm", "triton",
}
STATE = "The same card payment appears twice on my statement."
CHOICES = {
    "billing": "Investigate duplicate payments, charges and refunds.",
    "technical": "Investigate app crashes and sign-in failures.",
}
QUESTION = "Which support team should handle this request?"


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def forbidden_distribution(name):
    name = canonical(name)
    return name in FORBIDDEN_DISTRIBUTIONS or name.startswith(("nvidia-", "cuda-"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def remote_environment_required():
    if sys.platform != "linux" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Run this real-inference validation on remote Linux with Python 3.13")


def distribution_snapshot(python, run, log):
    script = (
        "import importlib.metadata,json,re; "
        "print(json.dumps({re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version "
        "for d in importlib.metadata.distributions()},sort_keys=True))"
    )
    run([str(python), "-c", script], log)
    return json.loads(Path(log).read_text().strip())


def installed_archive_sizes(pip_report, *, timeout=15, deadline=None):
    """Use exact wheel URLs from pip's report; do not redownload wheel bodies.

    pip's report omits byte sizes. A HEAD Content-Length supplies archive sizes,
    not total network transfer (which also includes metadata). Unknown sizes
    stay explicit instead of being silently counted as zero.
    """
    def inspect(item):
        info = item["download_info"]
        url = info["url"]
        parsed = urlparse(url)
        entry = {
            "name": canonical(item["metadata"]["name"]),
            "version": item["metadata"]["version"],
            "url": url,
            "sha256": info.get("archive_info", {}).get("hashes", {}).get("sha256"),
            "is_wheel": parsed.path.endswith(".whl"),
            "local_artifact": parsed.scheme == "file",
            "archive_bytes": None,
        }
        try:
            remaining = deadline - time.monotonic() if deadline is not None else timeout
            if remaining <= 0:
                raise TimeoutError("Archive measurement reached the overall deadline")
            if parsed.scheme == "file":
                entry["archive_bytes"] = Path(unquote(parsed.path)).stat().st_size
            elif parsed.scheme in {"http", "https"}:
                request = Request(url, method="HEAD", headers={"Accept-Encoding": "identity"})
                with urlopen(request, timeout=min(timeout, remaining)) as response:
                    length = response.headers.get("Content-Length")
                    if length is not None:
                        entry["archive_bytes"] = int(length)
            if entry["archive_bytes"] is None:
                entry["size_error"] = "No archive Content-Length was available"
        except Exception as error:
            entry["size_error"] = f"{type(error).__name__}: {error}"
        return entry

    with ThreadPoolExecutor(max_workers=4) as pool:
        artifacts = list(pool.map(inspect, pip_report["install"]))
    complete = all(item["archive_bytes"] is not None and item["is_wheel"] for item in artifacts)
    return {
        "artifacts": artifacts,
        "all_install_artifacts_are_wheels": all(item["is_wheel"] for item in artifacts),
        "wheel_archive_bytes_complete": complete,
        "total_wheel_archive_bytes": sum(item["archive_bytes"] for item in artifacts) if complete else None,
        "remote_wheel_archive_bytes": sum(item["archive_bytes"] for item in artifacts if not item["local_artifact"]) if complete else None,
        "known_archive_bytes": sum(item["archive_bytes"] or 0 for item in artifacts),
        "measurement_note": "Compressed archive sizes from pip report URLs and HTTP HEAD; includes the mounted candidate wheel in total, excludes pip metadata/network overhead and extracted install size.",
    }


def timing_summary(samples):
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "sample_p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "count": len(samples),
    }


def inference_worker(args):
    """Runs only inside the newly installed venv, in a fresh subprocess."""
    remote_environment_required()
    import importlib.abc
    import importlib.metadata
    import resource

    report = {"status": "running", "stage": args.worker, "mode": args.mode}
    attempts = []

    class NoOptionalImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in FORBIDDEN_IMPORTS:
                attempts.append(fullname)
                # In colab mode this is also a deliberately broken torchvision
                # sentinel. The real installed wheel remains intact on disk.
                raise RuntimeError(f"Unexpected optional runtime import: {fullname}")

    sys.meta_path.insert(0, NoOptionalImports())
    try:
        began = time.perf_counter()
        import gemmadecision
        from gemmadecision import constants
        report["package_import_seconds"] = time.perf_counter() - began
        assert constants.ONNX_REVISION and constants.ONNX_MANIFEST_SHA256, "Candidate wheel must contain published immutable ONNX pins"
        report["installed_version"] = importlib.metadata.version("gemmadecision")
        report["model_pins"] = {
            "source_revision": constants.MODEL_REVISION,
            "onnx_repository": constants.ONNX_REPO,
            "onnx_revision": constants.ONNX_REVISION,
            "onnx_manifest_sha256": constants.ONNX_MANIFEST_SHA256,
        }
        if args.worker == "cold":
            began = time.perf_counter()
            # Reproduce the user's exact failing public-API example before
            # any other model call; only check the output contract, not quality.
            example_choice = gemmadecision.decide("I was charged twice", choices=["billing", "technical"])
            report["download_load_first_decide_seconds"] = time.perf_counter() - began
            assert example_choice in {"billing", "technical"}
            report["reported_user_example"] = {
                "state": "I was charged twice",
                "choices": ["billing", "technical"],
                "result": example_choice,
                "inference_completed": True,
                "legal_choice": True,
                "accuracy_measured": False,
            }
            report["first_decide_fixture"] = "reported_user_example"
            from gemmadecision.simple import get_engine
            engine = get_engine()
            decide = lambda: gemmadecision.decide(STATE, choices=CHOICES, question=QUESTION)
            rank = lambda: gemmadecision.rank(STATE, choices=CHOICES, question=QUESTION)
            choice = decide()
        else:
            began = time.perf_counter()
            engine = gemmadecision.DecisionEngine.from_pretrained(offline=True)
            report["cached_offline_model_load_seconds"] = time.perf_counter() - began
            decide = lambda: engine.decide(STATE, CHOICES, question=QUESTION).choice
            rank = lambda: engine.rank(STATE, CHOICES, question=QUESTION)
            began = time.perf_counter()
            choice = decide()
            report["first_decide_after_cached_load_ms"] = (time.perf_counter() - began) * 1000
        assert choice in CHOICES
        assert engine.backend.metadata["backend"] == "onnx"
        assert engine.backend.metadata["device"] == "cpu"
        assert engine.backend.metadata["verified"] is True
        assert engine.cache_size == 0, "Latency measurements require the score cache to remain disabled"
        ranking = rank()
        assert {item.candidate for item in ranking.ranked} == set(CHOICES)
        assert ranking.ranked[0].candidate == choice
        assert all(math.isfinite(item.score) for item in ranking.ranked)
        assert math.isclose(sum(item.probability for item in ranking.ranked), 1.0, abs_tol=1e-6)
        report["choice"] = choice
        report["ranking"] = ranking.model_dump(mode="json")
        report["backend"] = engine.backend.metadata
        from gemmadecision.engine import render_state
        rendered = render_state(STATE, QUESTION)
        report["fixture"] = {
            "state": STATE, "choices": CHOICES, "question": QUESTION,
            "joint_tokens_per_candidate": [engine.backend.count_tokens(rendered + "\n\nCandidate action:\n" + text) for text in CHOICES.values()],
            "accuracy_measured": False,
        }
        report["warmup_calls_per_method"] = 3
        for name, function in [("decide", decide), ("rank", rank)]:
            for _ in range(3):
                function()
            samples = []
            for _ in range(10):
                began = time.perf_counter()
                function()
                samples.append((time.perf_counter() - began) * 1000)
            report[f"warm_{name}"] = timing_summary(samples)
        loaded = sorted(name for name in sys.modules if name.split(".")[0] in FORBIDDEN_IMPORTS)
        assert not loaded and not attempts, f"Unexpected framework imports: {loaded or attempts}"
        report["forbidden_import_attempts"] = attempts
        report["forbidden_loaded_modules"] = loaded
        report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        report["forbidden_import_attempts"] = attempts
        raise
    finally:
        write_json(args.output, report)


def validate_release(args):
    remote_environment_required()
    wheel = args.wheel.resolve() if args.wheel else None
    if wheel is not None and (not wheel.is_file() or wheel.suffix != ".whl"):
        raise ValueError("--wheel must be the built candidate wheel mounted in the remote container")
    output = args.output.resolve()
    artifacts = output.parent / (output.stem + "-artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "mode": args.mode,
        "installation_source": "candidate_wheel" if wheel else "public_pypi",
        "wheel": {"name": wheel.name, "sha256": sha256(wheel), "bytes": wheel.stat().st_size} if wheel else None,
        "requested_package_version": args.package_version,
        "artifacts_directory": str(artifacts),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "harness_sha256": sha256(__file__),
        "environment": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count()},
        "scope": "Fresh ordinary pip install and CPU functional/latency checks; no accuracy benchmark. Colab mode simulates existing Torch/vision packages, not the complete hosted Colab image.",
    }
    started = time.monotonic()
    deadline = started + args.timeout
    try:
        with tempfile.TemporaryDirectory(prefix="gemmadecision-release-") as temporary:
            workspace = Path(temporary)
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment.pop("PYTHONHOME", None)
            environment.pop("GEMMADECISION_MODEL_PATH", None)
            environment.update({
                "HF_HOME": str(workspace / "hf-cache"), "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_CACHE_DIR": "1", "PYTHONNOUSERSITE": "1",
            })
            environment.pop("HF_HUB_OFFLINE", None)

            def run(command, log, *, extra_env=None):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Release validation exceeded its bounded deadline")
                print("Running:", " ".join(map(str, command)), flush=True)
                began = time.monotonic()
                with Path(log).open("w") as stream:
                    subprocess.run(command, cwd=workspace, env={**environment, **(extra_env or {})},
                                   stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=remaining)
                return time.monotonic() - began

            report["venv_creation_seconds"] = run([
                sys.executable, "-m", "venv", str(workspace / "venv"),
            ], artifacts / "venv-creation.log")
            python = workspace / "venv/bin/python"
            initial = distribution_snapshot(python, run, artifacts / "initial-distributions.json")
            assert not any(forbidden_distribution(name) for name in initial)
            if args.mode == "colab":
                setup_report = artifacts / "preexisting-torch-pip-report.json"
                report["preexisting_torch_install_seconds"] = run([
                    str(python), "-m", "pip", "install", "--report", str(setup_report),
                    "--index-url", "https://download.pytorch.org/whl/cpu",
                    "torch==2.11.0", "torchvision==0.26.0",
                ], artifacts / "preexisting-torch-install.log")
            before = distribution_snapshot(python, run, artifacts / "before-distributions.json")
            if args.mode == "colab":
                assert before["torch"].split("+")[0] == "2.11.0"
                assert before["torchvision"].split("+")[0] == "0.26.0"
                report["preexisting_packages"] = {name: before[name] for name in ["torch", "torchvision"]}

            install_report = artifacts / "candidate-pip-report.json"
            requirement = str(wheel) if wheel else f"gemmadecision=={args.package_version}"
            report["candidate_install_seconds"] = run([
                str(python), "-m", "pip", "install", "--report", str(install_report),
                "--index-url", "https://pypi.org/simple", requirement,
            ], artifacts / "candidate-install.log")
            after = distribution_snapshot(python, run, artifacts / "after-distributions.json")
            report["installed_distributions"] = after
            pip_report = json.loads(install_report.read_text())
            package_artifacts = [item for item in pip_report["install"] if canonical(item["metadata"]["name"]) == "gemmadecision"]
            assert len(package_artifacts) == 1, "Install report must identify the GemmaDecision artifact"
            package_artifact = package_artifacts[0]
            if wheel is None:
                package_url = urlparse(package_artifact["download_info"]["url"])
                assert package_url.scheme == "https" and package_url.hostname == "files.pythonhosted.org", "Public release must come from a PyPI-hosted artifact"
                assert package_artifact["metadata"]["version"] == args.package_version == after["gemmadecision"]
                assert not package_artifact.get("is_direct", False), "Public release must use ordinary package resolution"
            report["installed_package_artifact"] = {
                "version": package_artifact["metadata"]["version"],
                "url": package_artifact["download_info"]["url"],
                "sha256": package_artifact["download_info"].get("archive_info", {}).get("hashes", {}).get("sha256"),
            }
            installed_names = {canonical(item["metadata"]["name"]) for item in pip_report["install"]}
            assert not any(forbidden_distribution(name) for name in installed_names), "Candidate pulled optional framework/server/CUDA packages"
            if args.mode == "minimal":
                assert not any(forbidden_distribution(name) for name in after), "Minimal environment contains optional framework/server packages"
            else:
                assert {name: after.get(name) for name in ["torch", "torchvision"]} == report["preexisting_packages"], "Candidate installation replaced preexisting Torch/vision"
                report["preexisting_versions_unchanged"] = True
                report["broken_vision_sentinel"] = "Inference worker rejects every torchvision import with RuntimeError while retaining the real installed wheel. No optional framework import is allowed."
            report["pip_check_seconds"] = run([str(python), "-m", "pip", "check"], artifacts / "pip-check.log")
            # Account for wheel sizes in a subprocess too: its hard remaining-
            # time timeout bounds redirects and slow HTTP headers, not merely
            # individual socket operations inside urllib.
            size_report = artifacts / "archive-sizes.json"
            report["archive_measurement_seconds"] = run([
                str(python), str(Path(__file__).resolve()), "--worker", "archives",
                "--pip-report", str(install_report), "--output", str(size_report),
            ], artifacts / "archive-sizes.log")
            report["candidate_archive_sizes"] = json.loads(size_report.read_text())
            assert report["candidate_archive_sizes"]["all_install_artifacts_are_wheels"], "Candidate required a source build"

            for stage in ["cold", "cached"]:
                result_file = artifacts / f"{stage}-inference.json"
                report[f"{stage}_worker_seconds"] = run([
                    str(python), str(Path(__file__).resolve()), "--worker", stage,
                    "--mode", args.mode, "--output", str(result_file),
                ], artifacts / f"{stage}-inference.log", extra_env={"HF_HUB_OFFLINE": "1"} if stage == "cached" else None)
                result = json.loads(result_file.read_text())
                assert result["status"] == "passed"
                report[stage] = result
            report["pip_report_file"] = str(install_report)
            report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output, report)
        print(json.dumps({"status": report["status"], "mode": args.mode, "output": str(output), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--wheel", type=Path)
    source.add_argument("--package-version", help="Exact public PyPI release to validate instead of a mounted wheel")
    parser.add_argument("--mode", choices=["minimal", "colab"], default="minimal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=540, help="Overall command deadline, leaving time for a 600s remote function to save results")
    parser.add_argument("--worker", choices=["cold", "cached", "archives"], help=argparse.SUPPRESS)
    parser.add_argument("--pip-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker == "archives":
        remote_environment_required()
        if args.pip_report is None:
            parser.error("Archive worker requires its pip report")
        write_json(args.output, installed_archive_sizes(json.loads(args.pip_report.read_text()), deadline=time.monotonic() + args.timeout))
    elif args.worker:
        inference_worker(args)
    else:
        if args.wheel is None and args.package_version is None:
            parser.error("--wheel or --package-version is required")
        if args.package_version is not None and not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+|\.dev\d+)?", args.package_version):
            parser.error("--package-version must name one exact release, for example 0.2.0")
        if not 1 <= args.timeout <= 540:
            parser.error("--timeout must be between 1 and 540 seconds")
        validate_release(args)


if __name__ == "__main__":
    main()
