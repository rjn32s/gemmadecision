"""Construct the real Rust server wrapper without opening sockets or models."""
import json
import sys
from unittest.mock import Mock

import pytest

from gemmadecision import cli
from gemmadecision.engine import DecisionEngine


def test_serve_uses_valid_real_granian_constructor(monkeypatch):
    granian = pytest.importorskip("granian")
    instances = []
    # Only serve is replaced: an invalid constructor keyword must fail here,
    # before a cloud deployment, without actually binding or loading a model.
    monkeypatch.setattr(granian.Granian, "serve", lambda self: instances.append(self))
    model_loader = Mock(side_effect=AssertionError("The unit test must not load weights"))
    monkeypatch.setattr(DecisionEngine, "from_pretrained", model_loader)
    monkeypatch.setattr(sys, "argv", [
        "gemmadecision", "serve", "--port", "18700", "--device", "cpu",
        "--model-path", "/unused-test-model", "--cache-size", "0",
    ])
    monkeypatch.setenv("GEMMADECISION_CONFIG", "{}")
    monkeypatch.delenv("GEMMADECISION_API_KEY", raising=False)
    cli.main()
    assert len(instances) == 1
    assert isinstance(instances[0], granian.Granian)
    model_loader.assert_not_called()
    config = json.loads(__import__("os").environ["GEMMADECISION_CONFIG"])
    assert config["device"] == "cpu"
    assert config["model_path"] == "/unused-test-model"
    assert config["cache_size"] == 0


def test_network_bind_requires_key_or_explicit_opt_in(monkeypatch):
    granian = pytest.importorskip("granian")
    serve = Mock(side_effect=AssertionError("Rejected configuration must not serve"))
    monkeypatch.setattr(granian.Granian, "serve", serve)
    monkeypatch.setattr(sys, "argv", ["gemmadecision", "serve", "--host", "0.0.0.0"])
    monkeypatch.delenv("GEMMADECISION_API_KEY", raising=False)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    serve.assert_not_called()


def test_network_bind_with_key_constructs_real_granian(monkeypatch):
    granian = pytest.importorskip("granian")
    instances = []
    monkeypatch.setattr(granian.Granian, "serve", lambda self: instances.append(self))
    monkeypatch.setattr(sys, "argv", ["gemmadecision", "serve", "--host", "0.0.0.0"])
    monkeypatch.setenv("GEMMADECISION_API_KEY", "unit-test-key-not-a-real-secret")
    monkeypatch.setenv("GEMMADECISION_CONFIG", "{}")
    cli.main()
    assert len(instances) == 1
