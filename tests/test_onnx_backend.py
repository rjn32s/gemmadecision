"""No model inference: exercise the CPU contract with a fake session/tokenizer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gemmadecision.backends import onnx as backend


def make_export(path: Path, **manifest_changes):
    path.mkdir(exist_ok=True)
    files = {
        "model.onnx": b"synthetic graph, never executed",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": json.dumps({"add_bos_token": True, "add_eos_token": False, "padding_side": "left"}).encode(),
        "config.json": json.dumps({"max_position_embeddings": 32768, "pad_token_id": 0, "bos_token_id": 2}).encode(),
        "joint_config.json": json.dumps({"format": "gemmadecision-full-joint-v4", "max_state_tokens": 2048, "max_action_tokens": 768}).encode(),
    }
    for name, data in files.items():
        (path / name).write_bytes(data)
    manifest = {
        "format": backend.MANIFEST_FORMAT,
        "source_model_revision": backend.constants.MODEL_REVISION,
        "model_file": "model.onnx",
        "dtype": "float32",
        "max_sequence_length": 32768,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        **manifest_changes,
    }
    raw = json.dumps(manifest).encode()
    (path / backend.MANIFEST_NAME).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def fake_runtime(monkeypatch):
    sessions = []

    class Tokenizer:
        @classmethod
        def from_file(cls, path):
            assert Path(path).name == "tokenizer.json"
            return cls()

        def no_truncation(self):
            self.truncation = False

        def no_padding(self):
            self.padding = False

        def token_to_id(self, value):
            return {"<pad>": 0, "<bos>": 2}[value]

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is True
            assert self.padding is self.truncation is False
            return SimpleNamespace(ids=[2] + [ord(letter) + 3 for letter in text])

        def encode_batch(self, texts, *, add_special_tokens):
            return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    class Options:
        def __init__(self):
            self.settings = {}

        def add_session_config_entry(self, key, value):
            self.settings[key] = value

    class Session:
        def __init__(self, path, *, sess_options, providers):
            assert Path(path).name in {"model.onnx", "model.int8.onnx"}
            self.options, self.providers, self.calls = sess_options, providers, []
            self.output_override = None
            sessions.append(self)

        def get_inputs(self):
            return [SimpleNamespace(name=name, type="tensor(int64)", shape=["batch", "sequence"]) for name in ["input_ids", "attention_mask"]]

        def get_outputs(self):
            return [SimpleNamespace(name="scores", type="tensor(float)", shape=["batch"])]

        def run(self, outputs, feed):
            assert outputs == ["scores"]
            ids, mask = feed["input_ids"], feed["attention_mask"]
            assert ids.dtype == mask.dtype == np.int64
            assert ids.shape == mask.shape
            assert np.all(ids[:, 0] == 2)
            assert np.all(np.diff(mask, axis=1) <= 0)  # right padding only
            assert np.all(ids[mask == 0] == 0)
            self.calls.append((ids.copy(), mask.copy()))
            if self.output_override is not None:
                return [self.output_override(len(ids))]
            # Unequal lengths and distinct last tokens catch lost ordering/padding.
            last = ids[np.arange(len(ids)), mask.sum(axis=1) - 1]
            return [last.astype(np.float32) + mask.sum(axis=1).astype(np.float32) / 100]

    ort = SimpleNamespace(
        SessionOptions=Options,
        InferenceSession=Session,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=Tokenizer))
    return SimpleNamespace(sessions=sessions, tokenizer=Tokenizer, session=Session)


def make_backend(tmp_path, fake_runtime, **kwargs):
    digest = make_export(tmp_path)
    return backend.ONNXBackend(tmp_path, manifest_sha256=digest, **kwargs)


def test_import_does_not_load_tensor_or_framework_libraries():
    code = """
import importlib.abc, sys
forbidden = {'torch', 'transformers', 'pydantic_ai', 'onnxruntime', 'numpy', 'tokenizers'}
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in forbidden:
            raise AssertionError('Unexpected eager dependency: ' + name)
sys.meta_path.insert(0, Deny())
import gemmadecision.backends.onnx
assert not forbidden.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_manifest_digest_is_the_trust_anchor(tmp_path, monkeypatch):
    digest = make_export(tmp_path)
    monkeypatch.setattr(backend.constants, "ONNX_MANIFEST_SHA256", digest, raising=False)
    assert backend.verify_onnx_files(tmp_path)["model_file"] == "model.onnx"
    original = (tmp_path / backend.MANIFEST_NAME).read_text()
    (tmp_path / backend.MANIFEST_NAME).write_text(original + " ")
    with pytest.raises(ValueError, match="manifest integrity"):
        backend.verify_onnx_files(tmp_path)


def test_graph_bytes_cannot_be_replaced_under_valid_manifest(tmp_path, fake_runtime):
    digest = make_export(tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"altered graph")
    with pytest.raises(ValueError, match="integrity check failed for model.onnx"):
        backend.ONNXBackend(tmp_path, manifest_sha256=digest)
    assert fake_runtime.sessions == []


def test_missing_trust_pin_fails_closed(tmp_path, monkeypatch):
    make_export(tmp_path)
    monkeypatch.setattr(backend.constants, "ONNX_MANIFEST_SHA256", None, raising=False)
    with pytest.raises(ValueError, match="No trusted"):
        backend.verify_onnx_files(tmp_path)


@pytest.mark.parametrize("name", ["../escape.onnx", "/tmp/escape.onnx", "sub/model.onnx", "..\\escape.onnx", "C:model.onnx"])
def test_manifest_rejects_file_escape_even_without_hash_check(tmp_path, name):
    make_export(tmp_path, model_file=name)
    with pytest.raises(ValueError, match="plain filenames"):
        backend.verify_onnx_files(tmp_path, verify=False)


def test_every_listed_external_data_file_is_verified(tmp_path):
    digest = make_export(tmp_path)
    manifest = json.loads((tmp_path / backend.MANIFEST_NAME).read_text())
    (tmp_path / "model.onnx.data").write_bytes(b"external tensors")
    manifest["files"]["model.onnx.data"] = hashlib.sha256(b"external tensors").hexdigest()
    raw = json.dumps(manifest).encode()
    (tmp_path / backend.MANIFEST_NAME).write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    backend.verify_onnx_files(tmp_path, manifest_sha256=digest)
    (tmp_path / "model.onnx.data").write_bytes(b"replacement")
    with pytest.raises(ValueError, match="model.onnx.data"):
        backend.verify_onnx_files(tmp_path, manifest_sha256=digest)


def test_wrong_source_revision_is_refused(tmp_path):
    make_export(tmp_path, source_model_revision="other")
    with pytest.raises(ValueError, match="pinned GemmaDecision"):
        backend.verify_onnx_files(tmp_path, verify=False)


def test_cpu_configuration_and_tokenizer_special_tokens(tmp_path, fake_runtime):
    value = make_backend(tmp_path, fake_runtime)
    session = fake_runtime.sessions[0]
    assert session.providers == ["CPUExecutionProvider"]
    assert session.options.intra_op_num_threads == 4
    assert session.options.inter_op_num_threads == 1
    assert session.options.execution_mode == "sequential"
    assert session.options.settings["session.intra_op.allow_spinning"] == "0"
    assert value.count_tokens("ab") == 3
    assert value.count_tokens("") == 1
    assert value.max_state_tokens == 2048 and value.max_action_tokens == 768
    assert value.context_limit == 3072 and value.encoder_context_limit == 32768
    assert value.metadata["verified"] is True
    assert value.metadata["model_revision"] == backend.constants.MODEL_REVISION


@pytest.mark.parametrize("published,verify", [(False, True), (True, True), (True, False)])
def test_metadata_distinguishes_source_weights_from_verified_export(
    tmp_path, fake_runtime, monkeypatch, published, verify
):
    digest = make_export(tmp_path)
    revision = "a" * 40
    monkeypatch.setattr(backend.constants, "ONNX_REVISION", revision)
    monkeypatch.setattr(backend.constants, "ONNX_MANIFEST_SHA256", digest if published else "0" * 64)
    value = backend.ONNXBackend(tmp_path, manifest_sha256=digest, verify=verify)
    assert value.metadata["onnx_manifest_sha256"] == digest
    assert value.metadata["onnx_revision"] == (revision if published and verify else None)
    assert value.metadata["model_revision"] == (backend.constants.MODEL_REVISION if verify else None)
    assert value.metadata["verified"] is verify


def test_batches_bound_padded_memory_and_restore_input_order(tmp_path, fake_runtime):
    value = make_backend(tmp_path, fake_runtime, max_batch_tokens=12, max_batch_size=3)
    texts = ["abcde", "y", "pqrs", "x"]
    actual = value.score_pairs(texts)
    expected = [ord(text[-1]) + 3 + (len(text) + 1) / 100 for text in texts]
    assert actual == pytest.approx(expected)
    assert [ids.shape for ids, _ in value.session.calls] == [(2, 2), (2, 6)]
    assert all(ids.size <= 12 for ids, _ in value.session.calls)


def test_strict_uses_single_inputs_without_changing_order(tmp_path, fake_runtime):
    value = make_backend(tmp_path, fake_runtime, strict=True)
    result = value.score_pairs(["ab", "c"])
    assert result == pytest.approx([ord("b") + 3.03, ord("c") + 3.02])
    assert len(value.session.calls) == 2
    assert all(ids.shape[0] == 1 for ids, _ in value.session.calls)
    assert value.metadata["max_batch_size"] == 1


def test_invalid_or_overlong_batch_performs_no_inference(tmp_path, fake_runtime):
    value = make_backend(tmp_path, fake_runtime, max_batch_tokens=10)
    assert value.score_pairs([]) == []
    with pytest.raises(ValueError, match="max_batch_tokens"):
        value.score_pairs(["ok", "x" * 10])
    with pytest.raises(ValueError, match="serving limit"):
        value.score_pairs(["ok", "x" * 3072])
    with pytest.raises(TypeError):
        value.score_pairs("one string")
    with pytest.raises(ValueError):
        value.score_pairs(["ok", ""])
    with pytest.raises(TypeError):
        value.count_tokens(None)
    assert value.session.calls == []


@pytest.mark.parametrize("output", [lambda n: np.full(n, np.nan), lambda n: np.full(n, np.inf), lambda n: np.zeros((n, 2)), lambda n: np.zeros(())])
def test_invalid_graph_results_do_not_escape_as_scores(tmp_path, fake_runtime, output):
    value = make_backend(tmp_path, fake_runtime)
    value.session.output_override = output
    with pytest.raises(RuntimeError, match="non-finite or incorrectly shaped"):
        value.score_pairs(["ab", "c"])


@pytest.mark.parametrize("kwargs", [{"device": "cuda"}, {"max_batch_tokens": 0}, {"max_batch_size": 0}, {"num_threads": 0}, {"num_threads": 65}])
def test_invalid_runtime_settings_fail_before_session_creation(tmp_path, fake_runtime, kwargs):
    with pytest.raises(ValueError):
        make_backend(tmp_path, fake_runtime, **kwargs)
    assert fake_runtime.sessions == []


def test_mismatched_graph_contract_is_rejected(tmp_path, fake_runtime, monkeypatch):
    monkeypatch.setattr(fake_runtime.session, "get_inputs", lambda _: [SimpleNamespace(name="embeddings", type="tensor(float)", shape=["batch", "features"])])
    with pytest.raises(ValueError, match="input_ids and attention_mask"):
        make_backend(tmp_path, fake_runtime)


def test_eos_postprocessor_mismatch_is_rejected_before_graph_load(tmp_path, fake_runtime, monkeypatch):
    original = fake_runtime.tokenizer.encode
    def with_eos(self, text, **kwargs):
        value = original(self, text, **kwargs)
        value.ids.append(1)
        return value
    monkeypatch.setattr(fake_runtime.tokenizer, "encode", with_eos)
    with pytest.raises(ValueError, match="BOS/padding behavior"):
        make_backend(tmp_path, fake_runtime)
    assert fake_runtime.sessions == []
