"""CPU scoring with a verified, complete GemmaDecision ONNX graph.

Only NumPy, ONNX Runtime and Hugging Face tokenizers are needed at inference.
The graph includes the encoder, normalized last-token pooling and scalar head;
it returns raw scores, not probabilities. Importing this module is lightweight.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any

from .. import constants


MANIFEST_NAME = "onnx_manifest.json"
MANIFEST_FORMAT = "gemmadecision-onnx-v1"
MAX_SEQUENCE_LENGTH = 3072
_REQUIRED_FILES = {"tokenizer.json", "tokenizer_config.json", "config.json", "joint_config.json"}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_name(value: Any) -> str:
    # Flat filenames also avoid platform-specific drive and backslash escapes.
    if (
        not isinstance(value, str)
        or value in {"", ".", "..", MANIFEST_NAME}
        or any(character in value for character in "/\\\x00:")
        or Path(value).name != value
    ):
        raise ValueError("ONNX manifest files must use plain filenames in the model directory")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def verify_onnx_files(
    path: str | Path, *, manifest_sha256: str | None = None, verify: bool = True
) -> dict[str, Any]:
    """Bind graph/tokenizer bytes to a trusted manifest digest.

    A manifest downloaded beside a graph is not its own trust anchor. Normal
    callers use the package's immutable digest. An explicit digest supports
    trusted prepublication export checks. ``verify=False`` is for synthetic tests.
    No inference dependencies or executable Hub code are imported here.
    """
    root = Path(path)
    manifest_file = root / MANIFEST_NAME
    if not root.is_dir() or not manifest_file.is_file():
        raise ValueError(f"The ONNX model directory is incomplete: missing {MANIFEST_NAME}")
    if manifest_file.stat().st_size > 1024 * 1024:
        raise ValueError("ONNX manifest is too large")
    if verify:
        expected = manifest_sha256 or getattr(constants, "ONNX_MANIFEST_SHA256", None)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("No trusted ONNX manifest digest is configured for this release")
        if _digest(manifest_file) != expected:
            raise ValueError("ONNX manifest integrity check failed")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid ONNX manifest JSON") from error
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"This backend requires a {MANIFEST_FORMAT} export")
    if manifest.get("source_model_revision") != constants.MODEL_REVISION:
        raise ValueError("The ONNX export does not use the pinned GemmaDecision v0.4.0 weights")
    graph_name = _file_name(manifest.get("model_file"))
    if not graph_name.endswith(".onnx"):
        raise ValueError("model_file must name an ONNX graph")
    files = manifest.get("files")
    if not isinstance(files, dict) or not _REQUIRED_FILES.union({graph_name}).issubset(files):
        raise ValueError("ONNX manifest must list the graph and all inference configuration files")
    for name, expected in files.items():
        _file_name(name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid SHA256 for ONNX file {name}")
        file = root / name
        if not file.is_file():
            raise ValueError(f"The ONNX model directory is incomplete: missing {name}")
        if verify and _digest(file) != expected:
            raise ValueError(f"ONNX integrity check failed for {name}")
    return manifest


def _plan_batches(lengths: Sequence[int], max_tokens: int, max_size: int) -> list[list[int]]:
    if any(length < 1 for length in lengths):
        raise ValueError("Every joint input must have at least one token")
    if any(length > max_tokens for length in lengths):
        raise ValueError("A joint input exceeds max_batch_tokens; inputs are never truncated")
    batches: list[list[int]] = []
    current: list[int] = []
    for index in sorted(range(len(lengths)), key=lambda item: (lengths[item], item)):
        if current and (len(current) == max_size or lengths[index] * (len(current) + 1) > max_tokens):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


class ONNXBackend:
    """Reuse one CPU session with bounded right-padded joint-input batches.

    ``strict`` limits each forward to one input; it is not a claim of bitwise
    equivalence between ONNX and PyTorch kernels. Separate state/action limits
    are checked by DecisionEngine before the already-rendered strings arrive.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        max_batch_tokens: int = 8192,
        max_batch_size: int = 32,
        strict: bool = False,
        num_threads: int = 4,
        manifest_sha256: str | None = None,
        verify: bool = True,
    ) -> None:
        if device not in {"auto", "cpu"}:
            raise ValueError("The ONNX backend supports CPU only; select the Torch backend for CUDA or MPS")
        self.max_batch_tokens = _positive_int(max_batch_tokens, "max_batch_tokens")
        _positive_int(max_batch_size, "max_batch_size")
        self.num_threads = _positive_int(num_threads, "num_threads")
        if self.num_threads > 64:
            raise ValueError("num_threads must not exceed 64")
        self.max_batch_size = 1 if strict else max_batch_size
        self.strict = bool(strict)
        self.path = Path(model_path)
        self.manifest = verify_onnx_files(self.path, manifest_sha256=manifest_sha256, verify=verify)
        self.manifest_sha256 = _digest(self.path / MANIFEST_NAME)
        # A trusted prepublication/custom export can use the same source weights
        # without being the published ONNX artifact. Do not label it as that pin.
        self.onnx_revision = (
            getattr(constants, "ONNX_REVISION", None)
            if verify and self.manifest_sha256 == getattr(constants, "ONNX_MANIFEST_SHA256", None)
            else None
        )
        self.config = json.loads((self.path / "joint_config.json").read_text(encoding="utf-8"))
        encoder_config = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        tokenizer_config = json.loads((self.path / "tokenizer_config.json").read_text(encoding="utf-8"))
        if self.config.get("format") != "gemmadecision-full-joint-v4":
            raise ValueError("This backend requires a full joint GemmaDecision v4 export")
        self.max_state_tokens = _positive_int(self.config.get("max_state_tokens"), "max_state_tokens")
        self.max_action_tokens = _positive_int(self.config.get("max_action_tokens"), "max_action_tokens")
        if (self.max_state_tokens, self.max_action_tokens) != (2048, 768):
            raise ValueError("The frozen model requires state/action limits of 2048/768 tokens")
        self.encoder_context_limit = _positive_int(encoder_config.get("max_position_embeddings"), "encoder context limit")
        export_limit = _positive_int(self.manifest.get("max_sequence_length", MAX_SEQUENCE_LENGTH), "max_sequence_length")
        # A full causal graph has quadratic attention allocations. Keep direct
        # backend calls inside the same bounded serving contract as the API.
        self.context_limit = min(self.encoder_context_limit, export_limit, MAX_SEQUENCE_LENGTH)
        if tokenizer_config.get("add_bos_token") is not True or tokenizer_config.get("add_eos_token") is not False:
            raise ValueError("The frozen tokenizer must add BOS and omit automatic EOS")
        if encoder_config.get("pad_token_id") != 0 or encoder_config.get("bos_token_id") != 2:
            raise ValueError("The frozen tokenizer requires padding ID 0 and BOS ID 2")
        self.pad_token_id = 0
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as error:
            raise ImportError("Install CPU inference dependencies: pip install gemmadecision") from error
        self._np = np
        self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        if (
            self.tokenizer.token_to_id("<pad>") != 0
            or self.tokenizer.token_to_id("<bos>") != 2
            or self.tokenizer.encode("", add_special_tokens=True).ids != [2]
        ):
            raise ValueError("The tokenizer JSON does not preserve the frozen BOS/padding behavior")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Avoid idle worker spinning in applications that make occasional calls.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(self.path / self.manifest["model_file"]),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        if (
            {item.name for item in inputs} != {"input_ids", "attention_mask"}
            or any(item.type != "tensor(int64)" or len(item.shape) != 2 for item in inputs)
        ):
            raise ValueError("ONNX graph must accept int64 input_ids and attention_mask matrices")
        outputs = self.session.get_outputs()
        if len(outputs) != 1 or outputs[0].name != "scores" or outputs[0].type != "tensor(float)":
            raise ValueError("ONNX graph must return one FP32 scores output")
        self.device = "cpu"
        self.dtype = self.manifest.get("dtype", "float32")
        self.verified = bool(verify)
        self._forward_lock = threading.Lock()

    def count_tokens(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("Token-count input must be a string")
        with self._forward_lock:
            return len(self.tokenizer.encode(text, add_special_tokens=True).ids)

    def score_pairs(self, texts: Sequence[str]) -> list[float]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("Joint inputs must be a sequence of strings")
        if not texts:
            return []
        if any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("Every joint input must be a nonempty string")
        with self._forward_lock:
            rows = self.tokenizer.encode_batch(list(texts), add_special_tokens=True)
            tokens = [row.ids for row in rows]
            lengths = [len(row) for row in tokens]
            if any(length > self.context_limit for length in lengths):
                raise ValueError(f"A joint input exceeds the ONNX serving limit of {self.context_limit} tokens")
            batches = _plan_batches(lengths, self.max_batch_tokens, self.max_batch_size)
            scores = [0.0] * len(texts)
            np = self._np
            for indices in batches:
                longest = max(lengths[index] for index in indices)
                ids = np.full((len(indices), longest), self.pad_token_id, dtype=np.int64)
                mask = np.zeros_like(ids)
                for row, index in enumerate(indices):
                    length = lengths[index]
                    ids[row, :length] = tokens[index]
                    mask[row, :length] = 1
                output = self.session.run(["scores"], {"input_ids": ids, "attention_mask": mask})
                if len(output) != 1:
                    raise RuntimeError("ONNX backend returned an invalid score vector")
                values = np.asarray(output[0])
                if values.shape not in {(len(indices),), (len(indices), 1)} or not np.isfinite(values).all():
                    raise RuntimeError("ONNX backend returned non-finite or incorrectly shaped scores")
                for index, score in zip(indices, values.reshape(-1).tolist(), strict=True):
                    scores[index] = float(score)
            return scores

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "onnx",
            "device": self.device,
            "dtype": self.dtype,
            "head_dtype": "float32",
            "strict": self.strict,
            "num_threads": self.num_threads,
            "max_batch_tokens": self.max_batch_tokens,
            "max_batch_size": self.max_batch_size,
            "context_limit": self.context_limit,
            "encoder_context_limit": self.encoder_context_limit,
            "model_revision": constants.MODEL_REVISION if self.verified else None,
            "onnx_manifest_sha256": self.manifest_sha256,
            "onnx_revision": self.onnx_revision,
            "verified": self.verified,
        }
