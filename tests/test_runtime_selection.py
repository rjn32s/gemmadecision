"""Backend selection must be deterministic and precede any model download."""
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gemmadecision import cli
from gemmadecision import engine, model


@pytest.fixture
def runtime_stubs(monkeypatch, tmp_path):
    download = Mock(return_value=tmp_path)
    monkeypatch.setattr(model, "download_model", download)
    constructors = {}
    for name, class_name in [("onnx", "ONNXBackend"), ("torch", "TorchBackend"), ("vllm", "VLLMBackend")]:
        constructor = Mock(return_value=SimpleNamespace(name=name))
        constructors[name] = constructor
        monkeypatch.setitem(sys.modules, f"gemmadecision.backends.{name}", SimpleNamespace(**{class_name: constructor}))
    monkeypatch.setattr(engine, "find_spec", lambda name: object())
    return SimpleNamespace(download=download, constructors=constructors)


@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_default_is_onnx_without_querying_optional_frameworks(runtime_stubs, monkeypatch, device):
    monkeypatch.setattr(engine, "find_spec", Mock(side_effect=AssertionError("ONNX selection must not inspect Torch")))
    result = engine.DecisionEngine.from_pretrained(device=device, offline=True)
    assert result.backend.name == "onnx"
    runtime_stubs.download.assert_called_once_with(offline=True, backend="onnx")
    assert runtime_stubs.constructors["onnx"].call_args.kwargs["device"] == device
    runtime_stubs.constructors["torch"].assert_not_called()


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_explicit_accelerator_selects_torch(runtime_stubs, device):
    result = engine.DecisionEngine.from_pretrained(device=device)
    assert result.backend.name == "torch"
    runtime_stubs.download.assert_called_once_with(offline=False, backend="torch")


@pytest.mark.parametrize("has_onnx,expected", [(True, "onnx"), (False, "torch")])
def test_local_artifact_contract_selects_backend_without_download(runtime_stubs, tmp_path, has_onnx, expected):
    (tmp_path / "joint_config.json").write_text("{}")
    if has_onnx:
        (tmp_path / "onnx_manifest.json").write_text("{}")
    result = engine.DecisionEngine.from_pretrained(tmp_path)
    assert result.backend.name == expected
    runtime_stubs.download.assert_not_called()


@pytest.mark.parametrize("selected", ["onnx", "torch", "vllm"])
def test_explicit_backend_preserves_selection_and_batch_settings(runtime_stubs, selected):
    result = engine.DecisionEngine.from_pretrained(backend=selected, max_batch_size=7, max_batch_tokens=1000)
    assert result.backend.name == selected
    runtime_stubs.download.assert_called_once_with(offline=False, backend=selected)
    kwargs = runtime_stubs.constructors[selected].call_args.kwargs
    assert kwargs["max_num_seqs" if selected == "vllm" else "max_batch_size"] == 7


@pytest.mark.parametrize("kwargs", [
    {"backend": "vllm", "device": "cpu"},
    {"backend": "vllm", "device": "mps"},
    {"backend": "vllm", "strict": True},
    {"backend": "vllm", "gpu_memory_utilization": 1},
    {"backend": "onnx", "device": "cuda"},
    {"backend": "unknown"},
    {"device": "unknown"},
    {"max_batch_tokens": 0},
    {"max_batch_size": 0},
    {"cache_size": -1},
])
def test_invalid_configuration_fails_before_download(runtime_stubs, kwargs):
    with pytest.raises(ValueError):
        engine.DecisionEngine.from_pretrained(**kwargs)
    runtime_stubs.download.assert_not_called()
    for constructor in runtime_stubs.constructors.values():
        constructor.assert_not_called()


@pytest.mark.parametrize("selected,missing", [("torch", "torch"), ("torch", "transformers"), ("torch", "safetensors"), ("vllm", "vllm")])
def test_missing_optional_dependency_errors_before_any_download(runtime_stubs, monkeypatch, selected, missing):
    monkeypatch.setattr(engine, "find_spec", lambda name: None if name == missing else object())
    with pytest.raises(ImportError, match=rf"gemmadecision\[{selected}\]"):
        engine.DecisionEngine.from_pretrained(backend=selected)
    runtime_stubs.download.assert_not_called()
    runtime_stubs.constructors[selected].assert_not_called()


def test_default_engine_path_never_imports_optional_training_or_server_packages():
    code = """
import importlib.abc, sys
from types import SimpleNamespace
forbidden = {'torch','transformers','safetensors','vllm','pydantic_ai','granian','fastapi'}
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in forbidden:
            raise AssertionError('Unexpected import: ' + name)
sys.meta_path.insert(0, Deny())
from gemmadecision import DecisionEngine
from gemmadecision import model
import gemmadecision.backends.onnx as backend
model.download_model = lambda **kwargs: '/fake-model'
backend.ONNXBackend = lambda *args, **kwargs: SimpleNamespace(name='onnx')
assert DecisionEngine.from_pretrained().backend.name == 'onnx'
assert not forbidden.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("selected", [None, "torch", "vllm"])
def test_cli_download_selects_only_requested_artifact(runtime_stubs, monkeypatch, selected, tmp_path):
    args = ["gemmadecision", "download", "--output", str(tmp_path)]
    if selected:
        args += ["--backend", selected]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    runtime_stubs.download.assert_called_once_with(local_dir=tmp_path, backend=selected or "onnx")


def test_server_cli_passes_auto_without_loading_runtime(runtime_stubs, monkeypatch):
    serve = Mock()
    wrapper = Mock(return_value=SimpleNamespace(serve=serve))
    monkeypatch.setitem(sys.modules, "granian", SimpleNamespace(Granian=wrapper))
    monkeypatch.setitem(sys.modules, "granian.constants", SimpleNamespace(Interfaces=SimpleNamespace(ASGI="asgi")))
    monkeypatch.setattr(sys, "argv", ["gemmadecision", "serve"])
    monkeypatch.delenv("GEMMADECISION_API_KEY", raising=False)
    monkeypatch.setenv("GEMMADECISION_CONFIG", "{}")
    cli.main()
    import os
    assert json.loads(os.environ["GEMMADECISION_CONFIG"])["backend"] == "auto"
    runtime_stubs.download.assert_not_called()
    serve.assert_called_once()
