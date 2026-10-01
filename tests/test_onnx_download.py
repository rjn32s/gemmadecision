"""Pinned artifact downloads use a fake Hub and never access the network."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gemmadecision import constants, model


@pytest.fixture
def hub(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot" / "onnx"
    snapshot.mkdir(parents=True)
    artifacts = {
        "model.onnx": b"synthetic graph",
        "model.onnx.data": b"synthetic external tensors",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "config.json": b"{}",
        "joint_config.json": b"{}",
    }
    for name, data in artifacts.items():
        (snapshot / name).write_bytes(data)
    manifest = {
        "format": "gemmadecision-onnx-v1",
        "source_model_revision": constants.MODEL_REVISION,
        "model_file": "model.onnx",
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()},
    }
    def write_manifest(value):
        payload = json.dumps(value).encode()
        (snapshot / "onnx_manifest.json").write_bytes(payload)
        monkeypatch.setattr(constants, "ONNX_MANIFEST_SHA256", hashlib.sha256(payload).hexdigest())
    write_manifest(manifest)
    monkeypatch.setattr(constants, "ONNX_REVISION", "a" * 40)
    calls = []
    def fetch(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        assert repo == constants.ONNX_REPO
        assert filename.startswith("onnx/")
        return str(snapshot / filename.removeprefix("onnx/"))
    download = Mock(side_effect=fetch)
    snapshot_download = Mock(return_value=str(tmp_path / "original-snapshot"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download, snapshot_download=snapshot_download))
    return SimpleNamespace(root=tmp_path, snapshot=snapshot, artifacts=artifacts, manifest=manifest,
                           write_manifest=write_manifest, calls=calls, download=download,
                           snapshot_download=snapshot_download)


def test_default_fetches_only_manifest_listed_files_at_pinned_revision(hub):
    assert model.download_model() == hub.snapshot
    assert {call[1] for call in hub.calls} == {"onnx/onnx_manifest.json", *(f"onnx/{name}" for name in hub.artifacts)}
    assert hub.calls[0][1] == "onnx/onnx_manifest.json"
    for _, _, kwargs in hub.calls:
        assert kwargs == {"revision": "a" * 40, "token": False, "local_files_only": False}
    hub.snapshot_download.assert_not_called()


def test_offline_propagates_to_manifest_and_all_runtime_files(hub):
    model.download_model(offline=True)
    assert all(kwargs["local_files_only"] is True for _, _, kwargs in hub.calls)


def test_hash_mismatch_is_rejected_before_any_artifact_fetch(hub):
    (hub.snapshot / "onnx_manifest.json").write_bytes(b"modified")
    with pytest.raises(ValueError, match="manifest integrity"):
        model.download_model()
    assert len(hub.calls) == 1


@pytest.mark.parametrize("field", ["ONNX_REVISION", "ONNX_MANIFEST_SHA256"])
def test_unpublished_pin_fails_closed_before_hub_access(hub, monkeypatch, field):
    monkeypatch.setattr(constants, field, None)
    with pytest.raises(RuntimeError, match="no published ONNX export"):
        model.download_model()
    hub.download.assert_not_called()


@pytest.mark.parametrize("name", ["../escape.data", "/tmp/escape.data", "nested/file.data", "..\\escape.data", "C:file.data", "", "onnx_manifest.json"])
def test_malformed_manifest_filename_never_reaches_hub(hub, name):
    hub.manifest["files"][name] = "a" * 64
    hub.write_manifest(hub.manifest)
    with pytest.raises(ValueError, match="plain filenames"):
        model.download_model()
    assert len(hub.calls) == 1


@pytest.mark.parametrize("mutation", [
    lambda manifest: manifest.update(format="other"),
    lambda manifest: manifest.update(source_model_revision="other"),
    lambda manifest: manifest.update(files=[]),
    lambda manifest: manifest["files"].update({"model.onnx": "not-sha256"}),
    lambda manifest: manifest["files"].pop("tokenizer_config.json"),
])
def test_invalid_manifest_contract_is_rejected_before_artifact_fetch(hub, mutation):
    mutation(hub.manifest)
    hub.write_manifest(hub.manifest)
    with pytest.raises(ValueError):
        model.download_model()
    assert len(hub.calls) == 1


@pytest.mark.parametrize("name", ["model.onnx", "model.onnx.data", "tokenizer.json"])
def test_tampered_runtime_bytes_fail_before_output_directory_is_created(hub, name):
    (hub.snapshot / name).write_bytes(b"tampered")
    destination = hub.root / "result"
    with pytest.raises(ValueError, match="integrity check failed"):
        model.download_model(local_dir=destination)
    assert not destination.exists()


def test_verified_download_can_be_copied_to_complete_offline_directory(hub):
    destination = hub.root / "result"
    result = model.download_model(local_dir=destination)
    assert result == destination
    assert {path.name for path in destination.iterdir()} == {*hub.artifacts, "onnx_manifest.json"}
    for name, data in hub.artifacts.items():
        assert (destination / name).read_bytes() == data


def test_existing_destination_symlink_does_not_overwrite_another_file(hub):
    destination = hub.root / "result"
    destination.mkdir()
    outside = hub.root / "keep.txt"
    outside.write_text("preserve me")
    (destination / "model.onnx").symlink_to(outside)
    model.download_model(local_dir=destination)
    assert outside.read_text() == "preserve me"
    assert not (destination / "model.onnx").is_symlink()
    assert (destination / "model.onnx").read_bytes() == hub.artifacts["model.onnx"]


@pytest.mark.parametrize("selected", ["torch", "vllm"])
def test_original_model_download_uses_explicit_frozen_allowlist(hub, selected):
    from gemmadecision.backends.torch import FROZEN_FILES
    model.download_model(backend=selected, offline=True)
    hub.download.assert_not_called()
    kwargs = hub.snapshot_download.call_args.kwargs
    assert kwargs["revision"] == constants.MODEL_REVISION
    assert kwargs["token"] is False and kwargs["local_files_only"] is True
    assert set(kwargs["allow_patterns"]) == {*FROZEN_FILES, "LICENSE", "GEMMA_TERMS.txt", "GEMMA_PROHIBITED_USE_POLICY.txt"}
    assert not any("*" in name or name.endswith(".py") for name in kwargs["allow_patterns"])


def test_invalid_backend_never_reaches_hub(hub):
    with pytest.raises(ValueError, match="backend"):
        model.download_model(backend="invalid")
    hub.download.assert_not_called()
    hub.snapshot_download.assert_not_called()
