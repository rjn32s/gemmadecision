#!/usr/bin/env python3
"""Build and verify lossless compact storage for the frozen FP32 ONNX model.

Dedicated export host only. No trained weights, temperature, or source export
are changed. The default lossless16 path stores exactly representable weights
in FP16/BF16 and restores their original FP32 values before computation. The
scalar head and unsafe initializers stay FP32. No candidate is published.

--experimental-int8 enables unvalidated diagnostic candidates, capped at two.
The first INT8 export failed numerical fidelity; the next attempt correctly
stopped at a scalar-head preservation guard. ORT rewrites Gemm to MatMul before
applying exclusions, changing names and transposing initializers. The current
experimental guard remains deliberately strict and may refuse such rewrites.
These INT8 candidates are not the release artifact and must not be shipped.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import signal
import time

from export_onnx import (TOKENIZER_FILES, fixture_inputs, graph_info, make_manifest,
                         sha256, validate_runtime, write_json)


def walk_graphs(graph):
    """Include control-flow subgraphs without relying on exporter node names."""
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            # ONNX AttributeProto.GRAPH=5 and GRAPHS=10 are stable wire values.
            if attribute.type == 5:
                yield from walk_graphs(attribute.g)
            elif attribute.type == 10:
                for child in attribute.graphs:
                    yield from walk_graphs(child)


def head_tensor_fingerprints(model):
    """Bind every named scalar-head initializer to its exact values and type."""
    from onnx import numpy_helper

    result = {}
    for graph in walk_graphs(model.graph):
        for tensor in graph.initializer:
            if not tensor.name.startswith("head."):
                continue
            value = {"dtype": tensor.data_type, "shape": list(tensor.dims),
                     "sha256": hashlib.sha256(numpy_helper.to_array(tensor).tobytes()).hexdigest()}
            if tensor.name in result and result[tensor.name] != value:
                raise ValueError("Ambiguous scalar-head initializer name across subgraphs")
            result[tensor.name] = value
    if not result:
        raise ValueError("No scalar-head initializers identified; cannot certify preservation")
    return result


def head_quantization_exclusions(model, parameter_names):
    """Trace named head parameters through weight-only plumbing, recursively.

    Exporters can emit Linear as Gemm, MatMul+Add, or in If subgraphs. Zero
    top-level head MatMuls is valid; initializer preservation is verified after
    quantization independently of operator naming or the number of nodes.
    """
    aliases = set(parameter_names)
    nodes = [node for graph in walk_graphs(model.graph) for node in graph.node]
    transforms = {"Identity", "Cast", "Transpose", "Reshape", "Squeeze", "Unsqueeze"}
    while True:
        previous = len(aliases)
        for node in nodes:
            if node.op_type in transforms and any(name in aliases for name in node.input):
                aliases.update(node.output)
        if len(aliases) == previous:
            break
    excluded = []
    for node in nodes:
        if node.op_type in {"MatMul", "Gemm"} and (
            any(name in aliases for name in node.input)
            or "/head/" in node.name or node.name.startswith("head.")
        ):
            if not node.name:
                raise ValueError("An unnamed scalar-head matrix operation cannot be safely excluded")
            excluded.append(node.name)
    return sorted(set(excluded))


def verify_head_preserved(model, expected):
    observed = head_tensor_fingerprints(model)
    if observed != expected:
        raise ValueError("Quantization modified, removed, or replaced scalar-head initializer values")
    return {"initializer_count": len(expected), "values_bitwise_unchanged": True,
            "fingerprints": expected}


def compact_initializer(tensor, preferred="float16"):
    """Lossless storage only: every converted value must round-trip bitwise.

    FP16's exponent range is narrower than BF16's. Use BF16 when a complete
    initializer cannot round-trip through FP16, and preserve FP32 when neither
    format is exact. Computation still receives the original FP32 values.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    if tensor.data_type != onnx.TensorProto.FLOAT or math.prod(tensor.dims) < 32:
        return None, "retained"
    values = numpy_helper.to_array(tensor)
    original = values.view(np.uint32)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        half = values.astype(np.float16)
        restored = half.astype(np.float32)
    if preferred == "float16" and np.array_equal(restored.view(np.uint32), original):
        compact = numpy_helper.from_array(half.copy(), name=tensor.name + ".storage_fp16")
        return compact, "float16"
    # BF16 is the high 16 bits of an IEEE float32. This is exact only when all
    # discarded low bits are zero; there is no rounding or retraining here.
    if np.all((original & np.uint32(0xFFFF)) == 0):
        high = (original >> 16).astype("<u2")
        if not np.array_equal(high.astype(np.uint32) << 16, original):
            raise AssertionError("BF16 bitwise round-trip failed")
        compact = onnx.TensorProto()
        compact.name = tensor.name + ".storage_bf16"
        compact.data_type = onnx.TensorProto.BFLOAT16
        compact.dims.extend(tensor.dims)
        compact.raw_data = high.tobytes()
        return compact, "bfloat16"
    return None, "retained"


def lossless_storage(graph, embedding_only=False):
    import onnx

    initializers, casts = [], []
    converted, kept, saved = [], [], 0
    for tensor in graph.graph.initializer:
        if tensor.name.startswith("head."):
            initializers.append(tensor)
            kept.append({"name": tensor.name, "elements": math.prod(tensor.dims)})
            continue
        if embedding_only and tensor.name != "encoder.embed_tokens.weight":
            initializers.append(tensor)
            continue
        compact, dtype = compact_initializer(tensor)
        if compact is None:
            initializers.append(tensor)
            if tensor.data_type == onnx.TensorProto.FLOAT:
                kept.append({"name": tensor.name, "elements": math.prod(tensor.dims)})
            continue
        initializers.append(compact)
        casts.append(onnx.helper.make_node("Cast", [compact.name], [tensor.name],
                                           name=tensor.name + ".restore_fp32", to=onnx.TensorProto.FLOAT))
        elements = math.prod(tensor.dims)
        saved += elements * 2
        converted.append({"name": tensor.name, "elements": elements, "storage_dtype": dtype})
    if embedding_only and not converted:
        raise ValueError("Embedding did not support any lossless 16-bit storage")
    nodes = casts + list(graph.graph.node)
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(initializers)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    return {"converted": converted, "retained_fp32": kept,
            "initializer_bytes_saved": saved, "all_conversions_bitwise_lossless": True,
            "computation_dtype": "float32", "optimizer_may_expand_weights_at_load": True}


def rowwise_embedding(graph):
    """Standard ONNX Gather(INT8) -> Cast -> Mul(per-row scale).

    This quantizes the embedding independently for each token, avoiding the
    global range used in the failed initial per-tensor Gather candidate.
    No activation quantization or calibration examples are needed here.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    name = "encoder.embed_tokens.weight"
    tensor = next((item for item in graph.graph.initializer if item.name == name), None)
    if tensor is None or tensor.data_type != onnx.TensorProto.FLOAT or len(tensor.dims) != 2:
        raise ValueError("Expected the original float32 embedding initializer")
    consumers = [node for node in graph.graph.node if name in node.input]
    if len(consumers) != 1 or consumers[0].op_type != "Gather" or consumers[0].input[0] != name:
        raise ValueError("Embedding must feed exactly one Gather")
    node = consumers[0]
    if any(item.name == "axis" and item.i != 0 for item in node.attribute):
        raise ValueError("Expected token embedding Gather on axis zero")
    values = numpy_helper.to_array(tensor)
    scales = np.max(np.abs(values), axis=1, keepdims=True).astype(np.float32) / np.float32(127)
    scales[scales == 0] = 1.0
    quantized = np.clip(np.rint(values / scales), -127, 127).astype(np.int8)
    difference = np.abs(quantized.astype(np.float32) * scales - values)
    report = {"format": "symmetric_per_token_row_int8", "scale_axis": 0,
              "rows": int(values.shape[0]), "columns": int(values.shape[1]),
              "max_abs_weight_error": float(difference.max()),
              "mean_abs_weight_error": float(difference.mean()),
              "embedding_approximation": True}
    qname, sname = name + ".row_int8", name + ".row_scale"
    selected, floated, selected_scale = name + ".selected_int8", name + ".selected_float", name + ".selected_scale"
    replacement = [
        onnx.helper.make_node("Gather", [qname, node.input[1]], [selected], name=node.name + ".row_int8", axis=0),
        onnx.helper.make_node("Cast", [selected], [floated], name=node.name + ".to_fp32", to=onnx.TensorProto.FLOAT),
        onnx.helper.make_node("Gather", [sname, node.input[1]], [selected_scale], name=node.name + ".row_scale", axis=0),
        onnx.helper.make_node("Mul", [floated, selected_scale], list(node.output), name=node.name + ".dequantize_rows"),
    ]
    initializers = [item for item in graph.graph.initializer if item.name != name]
    initializers.extend([numpy_helper.from_array(quantized, qname), numpy_helper.from_array(scales, sname)])
    nodes = []
    for current in graph.graph.node:
        nodes.extend(replacement if current is node or current == node else [current])
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(initializers)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-path", "--onnx-path", dest="export_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-seconds", type=int, default=840)
    parser.add_argument("--skip-lossless", action="store_true", help="Skip regenerating the already validated lossless16 fallback")
    parser.add_argument("--experimental-int8", action="store_true", help="Opt into unvalidated INT8 diagnostics; known head guard failures remain fail-closed")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 64 or not 60 <= args.max_seconds <= 900:
        parser.error("Use 1–64 threads and a 60–900 second deadline")
    if args.skip_lossless and not args.experimental_int8:
        parser.error("--skip-lossless requires --experimental-int8; the default action builds the lossless release candidate")

    import numpy as np
    import onnx
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from gemmadecision.backends.onnx import verify_onnx_files

    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = args.export_path.resolve()
    manifest_digest = sha256(source / "onnx_manifest.json")
    manifest = verify_onnx_files(source, manifest_sha256=manifest_digest)
    if manifest["dtype"] != "float32" or manifest["model_file"] != "model.onnx":
        raise ValueError("Optimization requires the verified original FP32 export")
    original_report = json.loads((source / "export-report.json").read_text())
    if original_report.get("status") != "complete" or original_report["fp32_graph"]["sha256"] != manifest["files"]["model.onnx"]:
        raise ValueError("Export report does not bind to this complete FP32 graph")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    config = AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    cases, rows, batches = fixture_inputs(tokenizer)
    if [len(row) for row in rows] != original_report["validation_token_lengths"]:
        raise ValueError("Parity fixtures differ from the original export")
    reference = original_report["reference_scores"]
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "onnx", "onnxruntime", "safetensors", "tokenizers")}
    report = {"status": "running", "source_manifest_sha256": manifest_digest,
              "source_graph_sha256": manifest["files"]["model.onnx"],
              "script_sha256": sha256(Path(__file__)), "versions": versions,
              "threads": args.threads, "max_seconds": args.max_seconds, "candidates": {},
              "experimental_int8_requested": args.experimental_int8,
              "selection": "No automatic promotion; fixture passes require development fidelity validation."}

    def save(stage):
        report["stage"] = stage
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "optimization-report.json", report)
        print(f"ONNX candidates: {stage} ({report['elapsed_seconds']:.1f}s)", flush=True)

    def alarm_handler(signum, frame):
        raise TimeoutError("ONNX candidate preparation deadline reached")

    previous_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(args.max_seconds)
    save("source and reference verified")
    temporary_base = output / "matmul-int8-working.onnx"

    def finish(name, graph, details, dtype, require_lossless=False):
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.onnx"
        item = {"status": "running", "transformation": details}
        report["candidates"][name] = item
        onnx.save_model(graph, str(path), save_as_external_data=False)
        del graph
        gc.collect()
        onnx.checker.check_model(str(path))
        item["graph"] = graph_info(path)
        save(f"{name}: graph written")
        item["parity"] = validate_runtime(path, reference, cases, rows, batches, config.pad_token_id, args.threads)
        item["fixture_top1_gate_passed"] = item["parity"]["top1_agreement_count"] == len(cases)
        item["fixture_fp32_numerical_gate_passed"] = bool(np.allclose(reference, item["parity"]["scores"], atol=5e-4, rtol=2e-4))
        if require_lossless and not item["fixture_fp32_numerical_gate_passed"]:
            raise RuntimeError("Lossless storage candidate failed original FP32 numerical parity")
        for filename in TOKENIZER_FILES:
            shutil.copy2(source / filename, directory / filename)
        candidate_manifest = make_manifest(directory, config, item["graph"], versions, "model.onnx", dtype)
        candidate_manifest["source_fp32_graph_sha256"] = manifest["files"]["model.onnx"]
        candidate_manifest["optimizer_sha256"] = sha256(Path(__file__))
        candidate_manifest["optimization"] = name
        write_json(directory / "onnx_manifest.json", candidate_manifest)
        item["manifest_sha256"] = sha256(directory / "onnx_manifest.json")
        item["status"] = "complete"
        save(f"{name}: fixture parity measured")

    try:
        # First produce the guaranteed-fidelity transport fallback. It converts
        # only exactly representable initializers; head and unsafe values stay
        # FP32, and every kernel receives FP32 values after restoration Casts.
        if not args.skip_lossless:
            graph = onnx.load(str(source / "model.onnx"))
            details = lossless_storage(graph)
            finish("lossless16", graph, details, "float32_compute_lossless_16bit_storage", require_lossless=True)
            del graph
            gc.collect()
        else:
            report["lossless16_skipped"] = "Previously validated artifact may be used; no previous candidate is modified."

        if not args.experimental_int8:
            report["status"] = "complete"
            save("lossless16 complete; experimental INT8 disabled")
            return 0

        from onnxruntime.quantization import QuantType, quantize_dynamic

        # Exclude identifiable scalar-head operations across all control-flow
        # subgraphs, then assert every head initializer is bitwise unchanged.
        graph = onnx.load(str(source / "model.onnx"), load_external_data=False)
        head_fingerprints = head_tensor_fingerprints(graph)
        excluded = head_quantization_exclusions(graph, head_fingerprints)
        report["scalar_head_preservation"] = {"excluded_nodes": excluded,
                                               "original_initializers": head_fingerprints}
        del graph
        gc.collect()
        quantize_dynamic(
            str(source / "model.onnx"), str(temporary_base),
            op_types_to_quantize=["MatMul"], nodes_to_exclude=excluded,
            per_channel=True, reduce_range=False, weight_type=QuantType.QInt8,
            use_external_data_format=False, extra_options={"MatMulConstBOnly": True},
        )
        save("dynamic INT8 linears prepared; embedding and scalar head preserved")
        graph = onnx.load(str(temporary_base))
        head_check = verify_head_preserved(graph, head_fingerprints)
        details = lossless_storage(graph, embedding_only=True)
        details["scalar_head_quantized"] = False
        details["excluded_scalar_head_nodes"] = excluded
        details["scalar_head_verification"] = head_check
        finish("matmul-int8-embed16", graph, details, "int8_dynamic_linears_lossless_16bit_embedding_fp32_head")
        del graph
        gc.collect()
        if report["candidates"]["matmul-int8-embed16"]["fixture_top1_gate_passed"]:
            graph = onnx.load(str(temporary_base))
            head_check = verify_head_preserved(graph, head_fingerprints)
            details = rowwise_embedding(graph)
            details["scalar_head_quantized"] = False
            details["excluded_scalar_head_nodes"] = excluded
            details["scalar_head_verification"] = head_check
            finish("matmul-int8-row-embedding", graph, details, "int8_dynamic_linears_rowwise_int8_embedding_fp32_head")
        else:
            # U8S8 can saturate on AVX2/AVX512 without VNNI. A bounded third
            # candidate uses ORT's documented reduced-range mitigation rather
            # than compounding a failed linear approximation with embeddings.
            quantize_dynamic(
                str(source / "model.onnx"), str(temporary_base),
                op_types_to_quantize=["MatMul"], nodes_to_exclude=excluded,
                per_channel=True, reduce_range=True, weight_type=QuantType.QInt8,
                use_external_data_format=False, extra_options={"MatMulConstBOnly": True},
            )
            graph = onnx.load(str(temporary_base))
            head_check = verify_head_preserved(graph, head_fingerprints)
            details = lossless_storage(graph, embedding_only=True)
            details.update(scalar_head_quantized=False, excluded_scalar_head_nodes=excluded,
                           scalar_head_verification=head_check,
                           reduce_range=True, reason="Check documented U8S8 saturation mitigation after full-range fixture failure")
            finish("matmul-int8-reduced-embed16", graph, details, "int8_reduced_dynamic_linears_lossless_16bit_embedding_fp32_head")
        del graph
        gc.collect()
        report["status"] = "complete"
        save("complete")
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        save("failed")
        raise
    finally:
        if temporary_base.exists():
            temporary_base.unlink()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
