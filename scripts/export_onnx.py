#!/usr/bin/env python3
"""Export the pinned full joint encoder for Torch-free CPU serving.

Run on the dedicated export host, not a developer laptop. Example::

    python scripts/export_onnx.py --model-path /models/full-model \
        --output /models/onnx-v1 --threads 4 --quantize-int8

The primary model.onnx is FP32. Optional model.int8.onnx is an unpromoted
quantization candidate: its measured agreement does not establish accuracy.
No network, training, publication, or model loading occurs on import.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time


MODEL_REVISION = "785d530221c990671f29976902540101bb9c7647"
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "config.json", "joint_config.json",
)
TEMPERATURE = 4.136820402388508


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_wrapper(encoder, head):
    """Use the original HF layers with traceable, exact additive masks.

    Transformers 5.17 accepts an already prepared mask dictionary. Avoid its
    vmap/mask factories during export; they otherwise introduce data-dependent
    tracing. All model layers, rotary embeddings, normalization, and scaling
    are the actual pretrained implementations, not reimplemented substitutes.
    """
    import torch

    if encoder.config.model_type != "gemma3_text":
        raise ValueError("Only the frozen Gemma3 text encoder is supported")
    if encoder.config.use_bidirectional_attention:
        raise ValueError("The published encoder uses causal attention")
    if encoder.config.sliding_window != 512:
        raise ValueError("Unexpected sliding window; refusing to alter the release")
    if any(params["rope_type"] != "default" for params in encoder.config.rope_parameters.values()):
        raise ValueError("Only the published default rotary embeddings are supported")

    class JointScores(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = head
            self.window = encoder.config.sliding_window

        def forward(self, input_ids, attention_mask):
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            query = positions[:, None]
            key = positions[None, :]
            # Exactly HF causal_mask_function and sliding_window_overlay:
            # key <= query; sliding layers additionally require key > q - 512.
            causal = key <= query
            valid_keys = attention_mask[:, None, None, :].to(torch.bool)
            full_allowed = causal[None, None, :, :] & valid_keys
            sliding_allowed = full_allowed & ((key > query - self.window)[None, None, :, :])
            minimum = torch.finfo(torch.float32).min
            full_mask = torch.where(full_allowed, 0.0, minimum)
            sliding_mask = torch.where(sliding_allowed, 0.0, minimum)
            hidden = self.encoder(
                input_ids=input_ids,
                attention_mask={"full_attention": full_mask, "sliding_attention": sliding_mask},
                position_ids=positions.unsqueeze(0),
                use_cache=False,
            ).last_hidden_state
            # Gather instead of advanced indexing keeps both B and S dynamic.
            last_indices = (attention_mask.sum(-1) - 1).reshape(-1, 1, 1)
            last_indices = last_indices.expand(-1, 1, hidden.shape[-1])
            last = hidden.gather(1, last_indices).squeeze(1).float()
            normalized = torch.nn.functional.normalize(last, p=2, dim=-1, eps=1e-12)
            return self.head(normalized).reshape(-1)

    return JointScores().eval().requires_grad_(False)


def fixture_inputs(tokenizer):
    """Small unlabeled parity fixtures; no train/test benchmark is consumed."""
    from gemmadecision.engine import render_state

    scenarios = [
        ("billing", "I was charged twice for one order.", [
            "Investigate the duplicate payment.", "Help reset the password.",
            "Review delivery tracking.", "Ask a human support agent to review the case.",
        ]),
        ("account", "My password reset link has expired and I cannot sign in.", [
            "Help recover access to the account.", "Explain the latest invoice.",
            "Track the parcel.", "Escalate to a human support agent.",
        ]),
        ("evidence", "Record: The meeting starts at 10 AM. Claim: The meeting starts at 2 PM.", [
            "The record supports the claim.", "The record contradicts the claim.",
            "The record does not provide enough evidence.",
        ]),
    ]
    question = "Which option best matches the supplied information?"
    # Check both sides of the 512-token sliding-attention boundary, with a
    # heterogeneous padded batch. This catches accidentally global attention.
    for minimum in (520, 1025):
        prefix = "Support handover: the agent recorded the existing case notes. "
        state = prefix
        while len(tokenizer(state)["input_ids"]) < minimum:
            state += prefix
        state += " Current request: the customer sees two identical charges for one order."
        scenarios.append((f"long-{minimum}", state, [
            "Investigate the duplicate payment.", "Help reset the account password.",
        ]))
    cases, rows = [], []
    for name, state, candidates in scenarios:
        rendered = render_state(state, question)
        texts = [rendered + "\n\nCandidate action:\n" + candidate for candidate in candidates]
        encoded = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)["input_ids"]
        cases.append({"id": name, "offset": len(rows), "count": len(encoded)})
        rows.extend(encoded)
    # A BOS-only row tests sequence length one without fabricating a label.
    rows.append([tokenizer.bos_token_id])
    # Different lengths and batch sizes from the export trace; padding is always
    # at the right. Each row belongs to exactly one numerical validation batch.
    batches = [[0], [1, 4, 7], [2, 3], [5, 6, 8, 9, 10], [11, 12], [13, 14], [15]]
    assert sorted(index for batch in batches for index in batch) == list(range(len(rows)))
    return cases, rows, batches


def padded_tensors(rows, indices, pad_token_id):
    import torch

    width = max(len(rows[index]) for index in indices)
    ids = torch.full((len(indices), width), pad_token_id, dtype=torch.int64)
    mask = torch.zeros_like(ids)
    for position, index in enumerate(indices):
        count = len(rows[index])
        ids[position, :count] = torch.tensor(rows[index], dtype=torch.int64)
        mask[position, :count] = 1
    return ids, mask


def reference_scores(encoder, head, input_ids, attention_mask):
    import torch

    hidden = encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    indices = attention_mask.sum(-1) - 1
    last = hidden[torch.arange(input_ids.shape[0]), indices].float()
    return head(torch.nn.functional.normalize(last, dim=-1)).reshape(-1)


def agreement(reference, observed, cases):
    import numpy as np

    expected, actual = np.asarray(reference), np.asarray(observed)
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError("Missing or non-finite scores during parity validation")
    differences = np.abs(expected - actual)
    per_case = []
    for case in cases:
        start, stop = case["offset"], case["offset"] + case["count"]
        left, right = expected[start:stop], actual[start:stop]
        pa = np.exp((left - left.max()) / TEMPERATURE)
        pb = np.exp((right - right.max()) / TEMPERATURE)
        pa, pb = pa / pa.sum(), pb / pb.sum()
        ordered = sorted(left, reverse=True)
        per_case.append({
            "id": case["id"], "reference_top1": int(left.argmax()),
            "observed_top1": int(right.argmax()),
            "top1_agrees": bool(left.argmax() == right.argmax()),
            "reference_margin": float(ordered[0] - ordered[1]),
            "max_abs_score_error": float(np.max(np.abs(left - right))),
            "max_abs_probability_error": float(np.max(np.abs(pa - pb))),
        })
    return {
        "pair_count": len(reference), "case_count": len(cases),
        "max_abs_score_error": float(differences.max()),
        "mean_abs_score_error": float(differences.mean()),
        "top1_agreement_count": sum(case["top1_agrees"] for case in per_case),
        "cases": per_case,
        "interpretation": "Numerical/ranking agreement on unlabeled fixtures; not an accuracy benchmark.",
    }


def run_torch_batches(score, rows, batches, pad_token_id):
    import torch

    values = [None] * len(rows)
    shapes = []
    with torch.inference_mode():
        for indices in batches:
            ids, mask = padded_tensors(rows, indices, pad_token_id)
            result = score(ids, mask).tolist()
            for index, value in zip(indices, result, strict=True):
                values[index] = float(value)
            shapes.append(list(ids.shape))
    return values, shapes


def validate_runtime(model_path, reference, cases, rows, batches, pad_token_id, threads):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    start = time.monotonic()
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    load_seconds = time.monotonic() - start
    expected_inputs = {"input_ids": "tensor(int64)", "attention_mask": "tensor(int64)"}
    if {item.name: item.type for item in session.get_inputs()} != expected_inputs:
        raise ValueError("Exported graph input contract is incorrect")
    outputs = session.get_outputs()
    if len(outputs) != 1 or outputs[0].name != "scores" or outputs[0].type != "tensor(float)":
        raise ValueError("Exported graph must produce one float32 scores vector")
    values = [None] * len(rows)
    timings = []
    for indices in batches:
        ids, mask = padded_tensors(rows, indices, pad_token_id)
        start = time.monotonic()
        result = session.run(["scores"], {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
        timings.append({"shape": list(ids.shape), "seconds": time.monotonic() - start})
        if result.shape != (len(indices),):
            raise ValueError("ONNX output lost dynamic batch shape")
        for index, value in zip(indices, result.tolist(), strict=True):
            values[index] = float(value)
    report = agreement(reference, values, cases)
    report.update({"load_seconds": load_seconds, "batch_timings": timings,
                   "scores": values, "providers": session.get_providers()})
    del session
    gc.collect()
    return report


def graph_info(path):
    import onnx

    # Shape/initializer metadata only: don't deserialize external data files.
    graph = onnx.load(str(path), load_external_data=False)
    operator_counts = dict(collections.Counter(node.op_type for node in graph.graph.node))
    largest = sorted(graph.graph.initializer, key=lambda item: math.prod(item.dims), reverse=True)[:5]
    info = {
        "bytes": path.stat().st_size,
        "sha256": sha256(path), "operator_counts": operator_counts,
        "opsets": {item.domain or "ai.onnx": item.version for item in graph.opset_import},
        "largest_initializers": [{"name": item.name, "shape": list(item.dims),
                                  "dtype": onnx.TensorProto.DataType.Name(item.data_type)} for item in largest],
        "external_data": [],
    }
    for item in graph.graph.initializer:
        if item.data_location == onnx.TensorProto.EXTERNAL:
            location = next(entry.value for entry in item.external_data if entry.key == "location")
            if Path(location).name != location:
                raise ValueError("External data must use flat relative file names")
            if location not in info["external_data"]:
                info["external_data"].append(location)
    info["total_bytes"] = info["bytes"] + sum((path.parent / name).stat().st_size for name in info["external_data"])
    del graph
    gc.collect()
    return info


def make_manifest(output, config, graph_report, versions, model_file="model.onnx", dtype="float32"):
    names = [model_file, *graph_report["external_data"], *TOKENIZER_FILES]
    return {
        "format": "gemmadecision-onnx-v1",
        "source_model_repository": "rajan2k/GemmaDecision-270M",
        "source_model_revision": MODEL_REVISION,
        "model_file": model_file,
        "dtype": dtype, "output_dtype": "float32",
        "inputs": {"input_ids": "int64[batch,sequence]", "attention_mask": "int64[batch,sequence]"},
        "outputs": {"scores": "float32[batch]"},
        "padding_side": "right", "pad_token_id": config.pad_token_id,
        "max_state_tokens": 2048, "max_action_tokens": 768,
        "context_limit": config.max_position_embeddings,
        "max_sequence_length": 3072,
        "pooling": "l2_normalized_last_nonpadding_token",
        "score_semantics": "raw scalar ranking score; no softmax or probability calibration in graph",
        "files": {name: sha256(output / name) for name in names},
        "total_file_bytes": sum((output / name).stat().st_size for name in names),
        "export_versions": versions,
        "exporter_sha256": sha256(Path(__file__)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--quantize-int8", action="store_true", help="Also make and measure an unpromoted dynamic INT8 candidate")
    parser.add_argument("--reuse-fp32", action="store_true", help="Validate an existing FP32 export before quantization; no graph replacement")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.opset < 17:
        parser.error("threads must be positive and opset at least 17")

    import numpy as np
    import onnx
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer
    from gemmadecision.backends.torch import FROZEN_FILES, verify_model_files

    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "export-report.json"
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "onnx", "onnxruntime", "safetensors", "tokenizers")}
    report = {"status": "running", "versions": versions, "python": platform.python_version(),
              "platform": platform.platform(), "threads": args.threads,
              "source_model_revision": MODEL_REVISION, "source_sha256": dict(FROZEN_FILES),
              "exporter_sha256": sha256(Path(__file__)), "opset": args.opset}
    write_json(report_path, report)

    def checkpoint(stage):
        report["stage"] = stage
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(report_path, report)
        print(f"ONNX export: {stage} ({report['elapsed_seconds']:.1f}s)", flush=True)

    try:
        verify_model_files(args.model_path)
        checkpoint("source verified")
        joint = json.loads((args.model_path / "joint_config.json").read_text())
        if joint["format"] != "gemmadecision-full-joint-v4":
            raise ValueError("Expected the published full joint v4 model")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = "right"
        encoder = AutoModel.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32, attn_implementation="sdpa",
        ).cpu().eval().requires_grad_(False)
        encoder.config.use_cache = False
        config = encoder.config
        hidden, width = joint["head_config"]["hidden"], joint["head_config"]["width"]
        head = torch.nn.Sequential(torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, width),
                                   torch.nn.GELU(), torch.nn.Linear(width, 1)).float().eval().requires_grad_(False)
        head.load_state_dict(load_file(str(args.model_path / joint["head_file"])), strict=True)
        cases, rows, batches = fixture_inputs(tokenizer)
        report["validation_token_lengths"] = [len(row) for row in rows]
        report["validation_cases"] = cases
        checkpoint("FP32 encoder loaded")
        baseline, shapes = run_torch_batches(
            lambda ids, mask: reference_scores(encoder, head, ids, mask), rows, batches, config.pad_token_id)
        report["reference_scores"] = baseline
        report["validation_shapes"] = shapes
        checkpoint("FP32 SDPA reference scored")

        # Only the attention kernel changes. Model parameters stay untouched.
        encoder.set_attn_implementation("eager")
        wrapper = make_wrapper(encoder, head)
        wrapped, _ = run_torch_batches(wrapper, rows, batches, config.pad_token_id)
        report["explicit_mask_parity"] = agreement(baseline, wrapped, cases)
        if not np.allclose(baseline, wrapped, atol=2e-4, rtol=2e-4):
            raise RuntimeError("Explicit-mask eager model differs from the published FP32 SDPA reference")
        checkpoint("explicit masks and pooling match reference")
        graph_path = output / "model.onnx"
        if args.reuse_fp32:
            if not graph_path.is_file():
                raise ValueError("--reuse-fp32 requires an existing model.onnx")
        else:
            # Trace on short padded inputs; validation exercises B=1,2,3,5 and
            # S=1 through >1025, so neither dimension may become constant.
            sample_ids, sample_mask = padded_tensors(rows, [0, 4], config.pad_token_id)
            with torch.inference_mode():
                torch.onnx.export(
                    wrapper, (sample_ids, sample_mask), str(graph_path),
                    input_names=["input_ids", "attention_mask"], output_names=["scores"],
                    dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                                  "attention_mask": {0: "batch", 1: "sequence"},
                                  "scores": {0: "batch"}},
                    opset_version=args.opset, export_params=True, do_constant_folding=True,
                    dynamo=False, external_data=False,
                )
        checkpoint("FP32 graph exported")
        # Release PyTorch weights before ORT loads and optimizes the large graph.
        del wrapper, encoder, head, tokenizer
        gc.collect()
        onnx.checker.check_model(str(graph_path))
        report["fp32_graph"] = graph_info(graph_path)
        report["fp32_parity"] = validate_runtime(graph_path, baseline, cases, rows, batches, config.pad_token_id, args.threads)
        if not np.allclose(baseline, report["fp32_parity"]["scores"], atol=5e-4, rtol=2e-4):
            raise RuntimeError("FP32 ONNX scores differ from the published FP32 SDPA reference")
        for filename in TOKENIZER_FILES:
            shutil.copy2(args.model_path / filename, output / filename)
        manifest = make_manifest(output, config, report["fp32_graph"], versions)
        write_json(output / "onnx_manifest.json", manifest)
        report["manifest_sha256"] = sha256(output / "onnx_manifest.json")
        checkpoint("FP32 runtime parity passed; manifest written")

        if args.quantize_int8:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            candidate = output / "model.int8.onnx"
            quantize_dynamic(
                str(graph_path), str(candidate), op_types_to_quantize=["MatMul", "Gather"],
                per_channel=True, reduce_range=False, weight_type=QuantType.QInt8,
                use_external_data_format=False,
                extra_options={"MatMulConstBOnly": True},
            )
            onnx.checker.check_model(str(candidate))
            report["int8_graph"] = graph_info(candidate)
            report["int8_parity"] = validate_runtime(candidate, baseline, cases, rows, batches, config.pad_token_id, args.threads)
            report["int8_is_default"] = False
            # The separate candidate manifest can be reviewed/promoted by the
            # release process. Never silently replace the FP32 default here.
            candidate_manifest = make_manifest(output, config, report["int8_graph"], versions,
                                               "model.int8.onnx", "int8_dynamic_matmul_gather")
            write_json(output / "onnx_manifest.int8.json", candidate_manifest)
            report["int8_manifest_sha256"] = sha256(output / "onnx_manifest.int8.json")
            checkpoint("INT8 candidate measured; FP32 default preserved")
        report["status"] = "complete"
        checkpoint("complete")
        print(json.dumps({"report": str(report_path), "manifest_sha256": report["manifest_sha256"],
                          "fp32_bytes": report["fp32_graph"]["total_bytes"],
                          "int8_bytes": report.get("int8_graph", {}).get("total_bytes")}), flush=True)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        checkpoint("failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
