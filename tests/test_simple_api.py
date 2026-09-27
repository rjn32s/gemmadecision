"""One-call Python API checks with stub engines only; no model downloads."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import gemmadecision
from gemmadecision import simple
from gemmadecision.engine import DecisionEngine


@pytest.fixture
def stub_engine(monkeypatch):
    engine = Mock()
    engine.decide.return_value = SimpleNamespace(choice="billing")
    engine.rank.return_value = SimpleNamespace(ranked=["detailed-result"])
    monkeypatch.setattr(simple, "_engine", engine)
    return engine


def test_decide_list_choices_returns_label(stub_engine):
    result = gemmadecision.decide("I was charged twice", choices=["billing", "technical"])
    assert result == "billing"
    stub_engine.decide.assert_called_once_with(
        "I was charged twice", {"billing": "billing", "technical": "technical"},
        question="Choose the best option.",
    )


def test_decide_named_choices_preserves_descriptions(stub_engine):
    choices = {"billing": "Investigate unexpected charges", "technical": "Fix a product problem"}
    assert gemmadecision.decide({"issue": "duplicate charge"}, choices, question="Choose a team") == "billing"
    stub_engine.decide.assert_called_once_with(
        {"issue": "duplicate charge"}, choices, question="Choose a team",
    )


def test_rank_returns_detailed_result(stub_engine):
    result = gemmadecision.rank("My card is missing", ["card", "cash"], question="Choose a team")
    assert result is stub_engine.rank.return_value
    stub_engine.rank.assert_called_once_with("My card is missing", ["card", "cash"], question="Choose a team")


def test_duplicate_list_fails_before_loading_model(monkeypatch):
    loader = Mock(side_effect=AssertionError("Model loading must not happen"))
    monkeypatch.setattr(simple, "get_engine", loader)
    with pytest.raises(ValueError, match="distinct"):
        gemmadecision.decide("state", ["same", "same"])
    loader.assert_not_called()


def test_lazy_default_engine_constructed_once_for_concurrent_calls(monkeypatch):
    engine = object()
    loader = Mock(return_value=engine)
    monkeypatch.setattr(simple, "_engine", None)
    monkeypatch.delenv("GEMMADECISION_MODEL_PATH", raising=False)
    monkeypatch.setattr(DecisionEngine, "from_pretrained", loader)
    with ThreadPoolExecutor(max_workers=8) as workers:
        values = list(workers.map(lambda _: simple.get_engine(), range(16)))
    assert all(value is engine for value in values)
    loader.assert_called_once()


def test_local_model_path_can_bypass_download(monkeypatch, tmp_path):
    loader = Mock(return_value=object())
    monkeypatch.setattr(simple, "_engine", None)
    monkeypatch.setenv("GEMMADECISION_MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(DecisionEngine, "from_pretrained", loader)
    simple.get_engine()
    assert str(loader.call_args.kwargs["model_path"]) == str(tmp_path)
