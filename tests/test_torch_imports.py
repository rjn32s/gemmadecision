"""Dependency failures need useful diagnostics without patching other libraries."""
import builtins
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gemmadecision.backends import torch as backend


def media_failure(module_name, exception):
    """Give a synthetic import failure a real traceback in that package."""
    namespace = {"__name__": module_name, "failure": exception}
    exec(compile("def load():\n    raise failure\n", "optional-package.py", "exec"), namespace)
    try:
        namespace["load"]()
    except Exception as error:
        return error


def lazy_wrapper(cause):
    try:
        raise ModuleNotFoundError("Could not import module 'Gemma3TextModel'") from cause
    except ModuleNotFoundError as error:
        return error


@pytest.fixture
def model_directory(tmp_path):
    (tmp_path / "joint_config.json").write_text(json.dumps({
        "format": "gemmadecision-full-joint-v4",
        "head_config": {"hidden": 2, "width": 2},
    }))
    return tmp_path


@pytest.fixture
def stub_dependencies(monkeypatch):
    torch = SimpleNamespace(device=lambda _: SimpleNamespace(type="cpu"), float32="float32")
    model = SimpleNamespace(from_pretrained=Mock())
    tokenizer = SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(pad_token_id=0)))
    loader = Mock()
    monkeypatch.setattr(backend, "_import_torch_dependencies", lambda: (torch, loader, model, tokenizer))
    return model, loader, tokenizer


@pytest.mark.parametrize("failure,package", [
    (RuntimeError("operator torchvision::nms does not exist"), "torchvision"),
    (media_failure("torchaudio._extension", OSError("libtorch_cuda.so: cannot open shared object file")), "torchaudio"),
    (media_failure("librosa.core.constantq", AttributeError("module 'numpy' has no attribute 'complex'")), "librosa"),
])
def test_lazy_media_import_failure_explains_repair_and_preserves_cause(
    model_directory, stub_dependencies, failure, package
):
    model, head_loader, _ = stub_dependencies
    wrapped = lazy_wrapper(failure)
    model.from_pretrained.side_effect = wrapped
    with pytest.raises(ImportError) as caught:
        backend.TorchBackend(model_directory, device="cpu", verify=False)
    message = str(caught.value)
    assert package in message
    assert f"python -m pip uninstall {package}" in message
    assert "Restart" in message
    assert "gemmadecision[torch]" in message
    assert caught.value.__cause__ is wrapped
    assert wrapped.__cause__ is failure
    head_loader.assert_not_called()


@pytest.mark.parametrize("failure", [
    RuntimeError("unexpected model tensor shape"),
    OSError("model.safetensors could not be read"),
    ImportError("a Transformers API changed"),
    ModuleNotFoundError("missing an unrelated kernel", name="some_kernel"),
])
def test_unrelated_model_errors_are_not_misreported_as_optional_dependencies(
    model_directory, stub_dependencies, failure
):
    model, _, _ = stub_dependencies
    model.from_pretrained.side_effect = failure
    with pytest.raises(type(failure)) as caught:
        backend.TorchBackend(model_directory, device="cpu", verify=False)
    assert caught.value is failure


def test_tokenizer_import_failure_is_diagnosed_before_model_loading(model_directory, stub_dependencies):
    model, head_loader, tokenizer = stub_dependencies
    failure = RuntimeError("operator torchvision::nms does not exist")
    tokenizer.from_pretrained.side_effect = failure
    with pytest.raises(ImportError, match="torchvision") as caught:
        backend.TorchBackend(model_directory, device="cpu", verify=False)
    assert caught.value.__cause__ is failure
    model.from_pretrained.assert_not_called()
    head_loader.assert_not_called()


@pytest.mark.parametrize("missing", ["torch", "safetensors.torch", "transformers"])
def test_missing_runtime_gives_extra_without_importing_or_modifying_packages(monkeypatch, missing):
    original = builtins.__import__
    failure = ModuleNotFoundError(f"No module named {missing}", name=missing)
    def fake_import(name, *args, **kwargs):
        if name == missing:
            raise failure
        if name in {"torch", "safetensors.torch", "transformers"}:
            return SimpleNamespace(load_file=Mock(), AutoModel=Mock(), AutoTokenizer=Mock())
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"gemmadecision\[torch\]") as caught:
        backend._import_torch_dependencies()
    assert caught.value.__cause__ is failure


def test_unrelated_import_error_keeps_original_exception(monkeypatch):
    original = builtins.__import__
    failure = ImportError("cannot import required Torch symbol")
    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise failure
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as caught:
        backend._import_torch_dependencies()
    assert caught.value is failure
