"""Create an encoder-only vLLM view without changing a trained tensor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

VLLM_VERSION = "0.30.0"
POOLER_CONFIG = {"pooling_type": "LAST", "use_activation": False}
_COPY_NAMES = (
    "model.safetensors", "tokenizer.json", "tokenizer.model",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "LICENSE", "GEMMA_PROHIBITED_USE_POLICY.txt", "NOTICE",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_vllm(model_path: str | Path, output: str | Path | None = None) -> Path:
    """Copy an encoder-only package, retaining the original model unchanged.

    vLLM's pooling adapter accepts the encoder's unprefixed tensor names. Its
    normal loader scans all safetensors, so the scalar head MUST stay outside
    this directory. The original model remains the source of the scalar head.
    Existing destinations are accepted only when all exported bytes match.
    """
    source = Path(model_path).expanduser().resolve()
    config = json.loads((source / "config.json").read_text())
    joint = json.loads((source / "joint_config.json").read_text())
    if config.get("model_type") != "gemma3_text":
        raise ValueError("vLLM export requires a Gemma 3 text encoder")
    if joint.get("format") != "gemmadecision-full-joint-v4":
        raise ValueError("vLLM export supports the full-joint v4 model format")
    if not (source / "model.safetensors").is_file():
        raise ValueError("Expected the v4 single-file model.safetensors encoder")
    if not (source / "tokenizer_config.json").is_file():
        raise ValueError("The source model is missing its tokenizer configuration")
    source_hashes = {name: _sha256(source / name) for name in _COPY_NAMES
                     if (source / name).is_file()}
    original_config_hash = _sha256(source / "config.json")
    identity = hashlib.sha256(json.dumps(
        {"files": source_hashes, "config": original_config_hash}, sort_keys=True
    ).encode()).hexdigest()
    if output is None:
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        destination = cache / "gemmadecision" / "vllm" / identity[:24]
    else:
        destination = Path(output).expanduser().resolve()
    if destination == source:
        raise ValueError("Export destination must differ from the trained model")
    config["architectures"] = ["Gemma3ForCausalLM"]
    config_bytes = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()
    export_hashes = dict(source_hashes)
    export_hashes["config.json"] = hashlib.sha256(config_bytes).hexdigest()
    manifest = {
        "format": "gemmadecision-vllm-encoder-v1",
        "source_identity": identity,
        "source_config_sha256": original_config_hash,
        "vllm_version": VLLM_VERSION,
        "runner": "pooling", "convert": "embed", "pooler_config": POOLER_CONFIG,
        "tensor_values_changed": False,
        "head_included": False,
        "sha256": export_hashes,
    }

    def verify_existing() -> None:
        record = destination / "gemmadecision-vllm.json"
        if not record.is_file() or json.loads(record.read_text()) != manifest:
            raise FileExistsError(f"Destination contains a different export: {destination}")
        if set(p.name for p in destination.glob("*.safetensors")) != {"model.safetensors"}:
            raise ValueError("vLLM encoder directory contains unexpected weight files")
        allowed = set(export_hashes) | {"gemmadecision-vllm.json"}
        if any(path.name not in allowed for path in destination.iterdir()):
            raise ValueError("vLLM encoder directory contains unexpected files")
        for name, expected in export_hashes.items():
            if not (destination / name).is_file() or _sha256(destination / name) != expected:
                raise ValueError(f"vLLM export integrity check failed for {name}")

    if destination.exists():
        verify_existing()
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gemmadecision-vllm-", dir=destination.parent))
    try:
        for name in source_hashes:
            shutil.copy2(source / name, staging / name)
        (staging / "config.json").write_bytes(config_bytes)
        (staging / "gemmadecision-vllm.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        try:
            staging.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            verify_existing()  # A concurrent process may have finished first.
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Complete local v4 model directory")
    parser.add_argument("--output", required=True, help="New encoder-only export directory")
    arguments = parser.parse_args()
    print(export_vllm(arguments.model, arguments.output))


if __name__ == "__main__":
    main()
