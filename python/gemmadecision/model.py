"""Download immutable public artifacts, never executable Hub code."""
from pathlib import Path
from .constants import MODEL_REPO, MODEL_REVISION


def download_model(*, local_dir: str | Path | None = None, offline: bool = False) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Install serving dependencies: pip install 'gemmadecision[serve]'") from exc
    return Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=local_dir,
                                  local_files_only=offline, token=False,
                                  allow_patterns=["*.safetensors", "*.json", "tokenizer.model", "LICENSE",
                                                  "GEMMA_TERMS.txt", "GEMMA_PROHIBITED_USE_POLICY.txt",
                                                  "evidence/calibration.json"]))

