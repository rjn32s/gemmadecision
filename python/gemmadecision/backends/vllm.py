"""Optional vLLM pooling backend for the *joint* GemmaDecision encoder.

No dependency, model, CUDA context, or network request is loaded at import.
This backend is experimental until its deployment-specific GPU parity passes.
"""
from __future__ import annotations

from importlib.metadata import version
import json
import math
from pathlib import Path
from threading import Lock

from ..export_vllm import POOLER_CONFIG, VLLM_VERSION, export_vllm


class VLLMBackend:
    """Batch complete state/candidate pairs in vLLM; run the trained head FP32.

    The decoder stays on CUDA. Its 640-element last-token vectors are returned
    to a tiny CPU FP32 head. This avoids depending on internal vLLM worker APIs
    and preserves the original FP32 normalization and scalar-head precision.
    """

    name = "vllm"

    def __init__(
        self,
        model_path: str | Path,
        *,
        export_path: str | Path | None = None,
        gpu_memory_utilization: float = 0.20,
        max_model_len: int = 3072,
        max_num_seqs: int = 64,
        enforce_eager: bool = True,
    ) -> None:
        try:
            installed = version("vllm")
        except Exception as error:
            raise ImportError("Install gemmadecision[vllm] on a supported Linux GPU host") from error
        if installed != VLLM_VERSION:
            raise RuntimeError(f"This backend requires vllm=={VLLM_VERSION}; found {installed}")
        if not 0 < gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        if max_num_seqs < 1 or max_model_len < 1:
            raise ValueError("vLLM sequence limits must be positive")
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from vllm import LLM, PoolingParams
        from vllm.config import PoolerConfig

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("The vLLM backend requires a CUDA GPU with bfloat16 support")
        self.path = Path(model_path).expanduser().resolve()
        from .torch import verify_model_files
        verify_model_files(self.path)
        config = json.loads((self.path / "joint_config.json").read_text())
        encoder_config = json.loads((self.path / "config.json").read_text())
        self.max_state_tokens = int(config["max_state_tokens"])
        self.max_action_tokens = int(config["max_action_tokens"])
        self.context_limit = min(int(encoder_config["max_position_embeddings"]), max_model_len)
        self.device = "cuda"
        self.dtype = "bfloat16"
        self.head_device = "cpu"
        self.hidden = int(config["head_config"]["hidden"])
        width = int(config["head_config"]["width"])
        head_file = config["head_file"]
        if not isinstance(head_file, str) or Path(head_file).name != head_file:
            raise ValueError("head_file must be a filename in the model directory")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden), torch.nn.Linear(self.hidden, width),
            torch.nn.GELU(), torch.nn.Linear(width, 1),
        ).to(device="cpu", dtype=torch.float32).eval().requires_grad_(False)
        weights = load_file(str(self.path / head_file), device="cpu")
        if any(not torch.isfinite(weight).all() for weight in weights.values()):
            raise ValueError("Non-finite scalar head weights")
        self.head.load_state_dict(weights, strict=True)
        self.encoder_path = export_vllm(self.path, export_path)
        self._pooling_params = PoolingParams(use_activation=False)
        self._lock = Lock()
        self.llm = LLM(
            model=str(self.encoder_path), tokenizer=str(self.encoder_path),
            runner="pooling", convert="embed", dtype="bfloat16",
            trust_remote_code=False, pooler_config=PoolerConfig(**POOLER_CONFIG),
            max_model_len=self.context_limit, max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager, enable_prefix_caching=False,
            tensor_parallel_size=1, seed=0,
        )

    def _tokens(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        return self.tokenizer(text, add_special_tokens=True, truncation=False,
                              padding=False)["input_ids"]

    def count_tokens(self, text: str) -> int:
        return len(self._tokens(text))

    def score_pairs(self, texts: list[str]) -> list[float]:
        """Score already-rendered joint pairs, preserving input order.

        Explicit token IDs ensure exactly the frozen tokenizer's BOS behavior.
        No chat template, truncation, probabilities, candidate-only embedding,
        or independently cached action representation is used.
        """
        if not texts:
            return []
        tokens = [self._tokens(text) for text in texts]
        for ids in tokens:
            if not ids or len(ids) > self.context_limit:
                raise ValueError(f"Joint input must contain 1–{self.context_limit} tokens")
        import torch
        with self._lock, torch.inference_mode():
            outputs = self.llm.embed(
                [{"prompt_token_ids": ids} for ids in tokens],
                pooling_params=self._pooling_params, use_tqdm=False,
            )
            if len(outputs) != len(tokens):
                raise RuntimeError("vLLM returned an unexpected number of embeddings")
            vectors = torch.tensor([item.outputs.embedding for item in outputs],
                                   dtype=torch.float32, device="cpu")
            if vectors.shape != (len(tokens), self.hidden) or not torch.isfinite(vectors).all():
                raise RuntimeError("vLLM returned invalid last-token hidden states")
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
            scores = self.head(vectors).flatten().tolist()
        if len(scores) != len(texts) or not all(math.isfinite(score) for score in scores):
            raise RuntimeError("Non-finite or malformed ranking scores")
        return scores

    score_texts = score_pairs

    @property
    def metadata(self) -> dict:
        from .torch import MODEL_REVISION
        return {"backend": "vllm", "vllm_version": VLLM_VERSION,
                "device": self.device, "dtype": self.dtype,
                "head_device": self.head_device, "head_dtype": "float32",
                "pooling_type": "LAST", "normalization": "float32_after_pooling",
                "context_limit": self.context_limit, "prefix_caching": False,
                "model_revision": MODEL_REVISION, "verified": True}
