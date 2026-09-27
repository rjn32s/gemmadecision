"""Batched inference for the published full joint GemmaDecision encoder.

Importing this module does not import torch, load weights, or use the network.
The caller renders the training-format strings and checks the separate state
and candidate token limits. This backend checks each complete pair's length.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Any, Sequence


MODEL_REPOSITORY = "rajan2k/GemmaDecision-270M"
MODEL_REVISION = "785d530221c990671f29976902540101bb9c7647"

# These hashes are bound to the published v0.4.0 commit, not downloaded from
# the same untrusted directory as the model. Loading never executes model code.
FROZEN_FILES = {
    "model.safetensors": "d7a3e291bfdfa7cd85b33a8a99ef81a4a7d3192e46c77253f3daf14dfd7d6b95",
    "joint_head.safetensors": "72ec4e7d1f0908ad684eeae250f0f70128aa4342b46174bf92a3c9851c5dd965",
    "joint_config.json": "d48f2b5ad5fbea6a3d6723017a6f6a114260a331f0b917b8008de14501091533",
    "config.json": "c1c64396b2939c76f0fa091aa07815b6e6f5ac1a60bfe3878993f0f575dd3ff3",
    "tokenizer.json": "7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c",
    "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
    "tokenizer_config.json": "94b03056ec5831e9021c3e3fe9db682778c4f2d081c4188f1fcb8b90541cf1cf",
    "special_tokens_map.json": "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397",
    "added_tokens.json": "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
}


def verify_model_files(path: str | Path) -> None:
    """Verify the files consumed by inference against the pinned release."""
    root = Path(path)
    for name, expected in FROZEN_FILES.items():
        file = root / name
        if not file.is_file():
            raise ValueError(f"The model directory is incomplete: missing {name}")
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(
                f"Model integrity check failed for {name}; expected the published "
                f"GemmaDecision v0.4.0 revision {MODEL_REVISION}"
            )


def plan_batches(
    lengths: Sequence[int], max_batch_tokens: int, max_batch_size: int
) -> list[list[int]]:
    """Bucket by length, bounding padded tokens and preserving all indices.

    Returned indices are an execution plan; callers restore the original order.
    The token bound counts padding (batch size times longest input), not merely
    the sum of the unpadded lengths. No truncation or dropping is permitted.
    """
    if max_batch_tokens < 1 or max_batch_size < 1:
        raise ValueError("Batch token and size limits must be positive")
    if any(length < 1 for length in lengths):
        raise ValueError("Every joint input must have at least one token")
    if any(length > max_batch_tokens for length in lengths):
        raise ValueError(
            "A joint input exceeds max_batch_tokens; increase the batch token "
            "budget to encode this input without truncation"
        )
    batches: list[list[int]] = []
    current: list[int] = []
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        if current and (
            len(current) == max_batch_size
            or lengths[index] * (len(current) + 1) > max_batch_tokens
        ):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


class TorchBackend:
    """Load one full encoder and score joint inputs using bounded batches.

    CUDA uses BF16 when supported; CPU and MPS use FP32. Scalar-head inputs and
    parameters always use FP32, matching the frozen reference. ``strict=True``
    uses one input per forward pass, matching its numerical batch shape. The
    faster default may produce small batch-shape differences around near ties.

    Instantiate once and reuse it. The lock prevents overlapping forwards from
    separate caller threads; the serving scheduler should coalesce requests
    into a single score_pairs call to obtain cross-request batching.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        max_batch_tokens: int = 8192,
        max_batch_size: int = 32,
        strict: bool = False,
        verify: bool = True,
    ) -> None:
        self.path = Path(model_path)
        if not self.path.is_dir():
            raise ValueError("model_path must be a complete local model directory")
        if max_batch_tokens < 1 or max_batch_size < 1:
            raise ValueError("Batch token and size limits must be positive")
        if verify:
            verify_model_files(self.path)
        self.config = json.loads((self.path / "joint_config.json").read_text())
        if self.config.get("format") != "gemmadecision-full-joint-v4":
            raise ValueError("This backend requires a full joint GemmaDecision v4 model")
        self.max_state_tokens = int(self.config.get("max_state_tokens", 2048))
        self.max_action_tokens = int(self.config.get("max_action_tokens", 768))
        dimensions = self.config.get("head_config", {"hidden": 640, "width": 512})
        hidden, width = int(dimensions["hidden"]), int(dimensions["width"])
        if min(self.max_state_tokens, self.max_action_tokens, hidden, width) < 1:
            raise ValueError("Token limits and head dimensions must be positive")
        head_file = self.config.get("head_file", "joint_head.safetensors")
        if not isinstance(head_file, str) or Path(head_file).name != head_file:
            raise ValueError("head_file must name a file in the model directory")
        try:
            import torch
            from safetensors.torch import load_file
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "Install the local inference dependencies: pip install 'gemmadecision[serve]'"
            ) from error

        self._torch = torch
        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "mps", "cuda"}:
            raise ValueError("Supported devices are cpu, mps and cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable on this machine")
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable on this machine")
        dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.dtype = str(dtype).removeprefix("torch.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False
        )
        # Last-token pooling below assumes right padding; do not inherit an
        # arbitrary tokenizer padding side from the calling environment.
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The model tokenizer must define a padding token")
        self.encoder = AutoModel.from_pretrained(
            self.path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=dtype,
            attn_implementation="sdpa",
        ).to(self.device).eval().requires_grad_(False)
        if self.encoder.config.hidden_size != hidden:
            raise ValueError("Scalar head width does not match the encoder hidden size")
        self.context_limit = getattr(self.encoder.config, "max_position_embeddings", None)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, 1),
        ).to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        weights = load_file(str(self.path / head_file), device="cpu")
        if any(not torch.isfinite(value).all() for value in weights.values()):
            raise ValueError("Non-finite scalar head weights")
        self.head.load_state_dict(weights, strict=True)
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = 1 if strict else max_batch_size
        self.strict = strict
        self.verified = verify
        self._forward_lock = threading.Lock()

    def count_tokens(self, text: str) -> int:
        """Count tokens including the same BOS/special tokens used for scoring."""
        if not isinstance(text, str):
            raise TypeError("Token-count input must be a string")
        return len(self.tokenizer(
            text, add_special_tokens=True, truncation=False, padding=False
        )["input_ids"])

    def score_pairs(self, texts: Sequence[str]) -> list[float]:
        """Score full training-format state/candidate strings, in input order.

        The caller must separately enforce max_state_tokens/max_action_tokens
        on the original fields before combining them. This method cannot recover
        those field boundaries reliably from arbitrary text. An empty list is
        allowed for scheduler convenience; empty individual inputs are not.
        """
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("Joint inputs must be a sequence of strings")
        if not texts:
            return []
        if any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("Every joint input must be a nonempty string")
        tokens = self.tokenizer(
            list(texts), add_special_tokens=True, truncation=False, padding=False
        )["input_ids"]
        lengths = [len(row) for row in tokens]
        if self.context_limit is not None and any(
            length > self.context_limit for length in lengths
        ):
            raise ValueError(f"A joint input exceeds the encoder limit of {self.context_limit} tokens")
        batches = plan_batches(lengths, self.max_batch_tokens, self.max_batch_size)
        scores = [0.0] * len(texts)
        # inference_mode is entered here because it is thread-local: a long-lived
        # worker may call the backend from different executor threads.
        with self._forward_lock, self._torch.inference_mode():
            for indices in batches:
                batch_scores = self._score_tokens([tokens[i] for i in indices])
                for index, score in zip(indices, batch_scores, strict=True):
                    scores[index] = score
        return scores

    def _score_tokens(self, tokens: Sequence[Sequence[int]]) -> list[float]:
        torch = self._torch
        batch = self.tokenizer.pad(
            {"input_ids": tokens}, padding=True, return_attention_mask=True,
            return_tensors="pt",
        )
        ids = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)
        hidden = self.encoder(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).last_hidden_state
        positions = mask.sum(-1) - 1
        last = hidden[torch.arange(len(ids), device=self.device), positions].float()
        vectors = torch.nn.functional.normalize(last, dim=-1)
        if not torch.isfinite(vectors).all():
            raise RuntimeError("Non-finite model encoding")
        scores = self.head(vectors).reshape(-1).cpu().tolist()
        if any(not math.isfinite(score) for score in scores):
            raise RuntimeError("Non-finite ranking score")
        return scores

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "torch",
            "device": str(self.device),
            "dtype": self.dtype,
            "head_dtype": "float32",
            "strict": self.strict,
            "max_batch_tokens": self.max_batch_tokens,
            "max_batch_size": self.max_batch_size,
            "model_revision": MODEL_REVISION if self.verified else None,
            "verified": self.verified,
        }
