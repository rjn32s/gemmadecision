"""Exporter semantics with tiny tensors only; never load/download model weights."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_onnx.py"
spec = importlib.util.spec_from_file_location("export_onnx_script", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class TinyEncoder(torch.nn.Module):
    """Records masks and provides distinguishable token vectors for pooling."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=False,
            sliding_window=512,
            rope_parameters={"full_attention": {"rope_type": "default"}},
        )
        self.seen = None

    def forward(self, input_ids, attention_mask, position_ids, use_cache):
        self.seen = (attention_mask, position_ids, use_cache)
        values = input_ids.float()
        hidden = torch.stack((values, values + 2, torch.ones_like(values)), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_explicit_masks_preserve_causality_window_boundary_and_padding():
    encoder = TinyEncoder()
    head = torch.nn.Linear(3, 1, bias=False)
    wrapper = exporter.make_wrapper(encoder, head)
    ids = torch.ones((2, 515), dtype=torch.int64)
    mask = torch.ones_like(ids)
    mask[1, 3:] = 0
    wrapper(ids, mask)
    masks, positions, use_cache = encoder.seen
    full, sliding = masks["full_attention"], masks["sliding_attention"]
    blocked = torch.finfo(torch.float32).min
    assert full.shape == sliding.shape == (2, 1, 515, 515)
    assert full.dtype == sliding.dtype == torch.float32
    assert full[0, 0, 512, 0] == 0
    assert sliding[0, 0, 512, 0] == blocked  # exactly outside the 512-token window
    assert sliding[0, 0, 512, 1] == 0
    assert sliding[0, 0, 511, 0] == 0
    assert full[0, 0, 511, 512] == sliding[0, 0, 511, 512] == blocked
    assert full[1, 0, 514, 3] == sliding[1, 0, 514, 3] == blocked
    assert full[1, 0, 2, 2] == sliding[1, 0, 2, 2] == 0
    assert positions.tolist() == [list(range(515))]
    assert use_cache is False


def test_last_nonpadding_pooling_normalizes_before_head_and_is_batch_independent():
    encoder = TinyEncoder()
    head = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 2.0, -1.0]]))
    wrapper = exporter.make_wrapper(encoder, head)
    ids = torch.tensor([[2, 7, 9], [2, 4, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    scores = wrapper(ids, mask)
    expected = torch.tensor([(9 + 2 * 11 - 1) / (9 ** 2 + 11 ** 2 + 1) ** .5,
                             (4 + 2 * 6 - 1) / (4 ** 2 + 6 ** 2 + 1) ** .5])
    torch.testing.assert_close(scores, expected)
    torch.testing.assert_close(scores[1:], wrapper(ids[1:, :2], mask[1:, :2]))
    assert scores.shape == (2,)
    assert wrapper(torch.tensor([[2]]), torch.tensor([[1]])).shape == (1,)


def test_wrapper_rejects_unpublished_attention_semantics():
    encoder = TinyEncoder()
    encoder.config.use_bidirectional_attention = True
    with pytest.raises(ValueError, match="causal"):
        exporter.make_wrapper(encoder, torch.nn.Linear(3, 1))


def test_onnx_dynamic_axes_with_tiny_encoder_when_export_dependencies_available(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    # This tests the wrapper's dynamic pooling and mask plumbing, not the real
    # transformer graph. The export script validates the real graph remotely.
    encoder = TinyEncoder()
    wrapper = exporter.make_wrapper(encoder, torch.nn.Linear(3, 1))
    destination = tmp_path / "tiny.onnx"
    ids = torch.tensor([[2, 4, 6], [2, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    torch.onnx.export(
        wrapper, (ids, mask), str(destination), dynamo=False, opset_version=18,
        input_names=["input_ids", "attention_mask"], output_names=["scores"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                      "attention_mask": {0: "batch", 1: "sequence"},
                      "scores": {0: "batch"}},
    )
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    for new_ids, new_mask in [
        (torch.tensor([[2]]), torch.tensor([[1]])),
        (torch.tensor([[2, 4, 6, 7], [2, 3, 0, 0], [2, 1, 5, 0]]),
         torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]])),
    ]:
        expected = wrapper(new_ids, new_mask)
        result = session.run(["scores"], {"input_ids": new_ids.numpy(), "attention_mask": new_mask.numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(result), expected)
