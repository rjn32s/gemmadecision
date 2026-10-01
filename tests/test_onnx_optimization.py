"""Graph guards with tiny structural fixtures; no model or ONNX runtime needed."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for name in ("export_onnx", "optimize_onnx_candidates"):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
optimizer = sys.modules["optimize_onnx_candidates"]


def node(op_type, name, inputs, outputs, attributes=()):
    return SimpleNamespace(op_type=op_type, name=name, input=inputs,
                           output=outputs, attribute=attributes)


def graph(nodes, initializers=()):
    return SimpleNamespace(node=nodes, initializer=initializers)


def test_scalar_head_gemm_without_any_head_matmul_is_supported():
    model = SimpleNamespace(graph=graph([
        node("MatMul", "encoder_projection", ["hidden", "encoder.weight"], ["encoded"]),
        node("Gemm", "arbitrary_dense_1", ["encoded", "head.1.weight", "head.1.bias"], ["h"]),
        node("Gemm", "arbitrary_dense_2", ["h", "head.3.weight", "head.3.bias"], ["score"]),
    ]))
    excluded = optimizer.head_quantization_exclusions(model, {"head.1.weight", "head.3.weight"})
    assert excluded == ["arbitrary_dense_1", "arbitrary_dense_2"]


def test_head_weight_aliases_are_followed_inside_control_flow_subgraphs():
    branch = graph([
        node("Transpose", "transpose_weight", ["head.1.weight"], ["transposed"]),
        node("MatMul", "unnamely_projection", ["features", "transposed"], ["branch_score"]),
    ])
    attribute = SimpleNamespace(type=5, g=branch)
    model = SimpleNamespace(graph=graph([
        node("If", "dynamic_pooling_branch", ["condition"], ["scores"], [attribute]),
    ]))
    assert optimizer.head_quantization_exclusions(model, {"head.1.weight"}) == ["unnamely_projection"]


def test_no_quantizable_head_operation_does_not_block_initializer_guard():
    model = SimpleNamespace(graph=graph([node("Identity", "head_passthrough", ["x"], ["score"])]))
    assert optimizer.head_quantization_exclusions(model, {"head.1.weight"}) == []


def test_head_fingerprint_guard_detects_a_single_modified_value(monkeypatch):
    np = pytest.importorskip("numpy")
    fake_onnx = SimpleNamespace(numpy_helper=SimpleNamespace(to_array=lambda tensor: tensor.values))
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    weight = SimpleNamespace(name="head.1.weight", data_type=1, dims=[1, 2],
                             values=np.array([[1., 2.]], dtype=np.float32))
    model = SimpleNamespace(graph=graph([], [weight]))
    fingerprint = optimizer.head_tensor_fingerprints(model)
    assert optimizer.verify_head_preserved(model, fingerprint)["values_bitwise_unchanged"]
    weight.values[0, 1] = np.nextafter(np.float32(2), np.float32(3))
    with pytest.raises(ValueError, match="initializer values"):
        optimizer.verify_head_preserved(model, fingerprint)


def test_skip_lossless_requires_explicit_experimental_int8_opt_in(tmp_path):
    with pytest.raises(SystemExit) as error:
        optimizer.main(["--export-path", str(tmp_path), "--output", str(tmp_path / "output"),
                        "--skip-lossless"])
    assert error.value.code == 2
