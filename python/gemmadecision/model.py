"""Download immutable public artifacts, never executable Hub code."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from .constants import MODEL_REPO, MODEL_REVISION


def download_model(*, local_dir: str | Path | None = None, offline: bool = False,
                   backend: str = "onnx") -> Path:
    if backend in {"auto", "onnx"}:
        return download_onnx_model(local_dir=local_dir, offline=offline)
    if backend not in {"torch", "vllm"}:
        raise ValueError("backend must be auto, onnx, torch or vllm")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Install gemmadecision to download its model artifacts") from exc
    from .backends.torch import FROZEN_FILES
    return Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=local_dir,
                                  local_files_only=offline, token=False,
                                  allow_patterns=[*FROZEN_FILES, "LICENSE", "GEMMA_TERMS.txt",
                                                  "GEMMA_PROHIBITED_USE_POLICY.txt"]))


def download_onnx_model(*, local_dir: str | Path | None = None, offline: bool = False) -> Path:
    """Fetch only the pinned runtime files, without a training framework."""
    from .constants import ONNX_REPO, ONNX_REVISION, ONNX_SUBFOLDER, ONNX_MANIFEST_SHA256
    if not ONNX_REVISION or not ONNX_MANIFEST_SHA256:
        raise RuntimeError("This development build has no published ONNX export")
    from huggingface_hub import hf_hub_download
    from .backends.onnx import MANIFEST_NAME, MANIFEST_FORMAT, _file_name, verify_onnx_files
    def fetch(name):
        return Path(hf_hub_download(ONNX_REPO, f"{ONNX_SUBFOLDER}/{name}",
                    revision=ONNX_REVISION, token=False, local_files_only=offline))
    manifest_path = fetch(MANIFEST_NAME)
    if manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("ONNX manifest is too large")
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != ONNX_MANIFEST_SHA256:
        raise ValueError("ONNX manifest integrity check failed")
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("Invalid ONNX artifact manifest format")
    if manifest.get("source_model_revision") != MODEL_REVISION:
        raise ValueError("ONNX export does not use the pinned source model revision")
    graph = _file_name(manifest.get("model_file"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Invalid ONNX artifact manifest")
    if not graph.endswith(".onnx") or not {graph, "tokenizer.json", "tokenizer_config.json", "config.json", "joint_config.json"}.issubset(files):
        raise ValueError("ONNX manifest must list the graph and all inference configuration files")
    for name, digest in files.items():
        _file_name(name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA256 for ONNX file {name}")
    with ThreadPoolExecutor(max_workers=4) as workers:
        paths = list(workers.map(fetch, files))
    # Verify in the Hub snapshot before returning it or touching an output
    # directory. A successful `download` command promises usable verified files.
    verify_onnx_files(manifest_path.parent, manifest_sha256=ONNX_MANIFEST_SHA256)
    if local_dir is None:
        return manifest_path.parent
    destination = Path(local_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for path in [*paths, manifest_path]:
        target = destination / path.name
        if path.resolve() != target.resolve():
            # Atomic replacement also avoids following an existing destination
            # symlink and overwriting a file outside the requested directory.
            with tempfile.NamedTemporaryFile(dir=destination, prefix=".gemmadecision-", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                shutil.copy2(path, temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return destination
