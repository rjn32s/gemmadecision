"""Exercise the runnable cookbook through real public APIs and a fake encoder."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from gemmadecision.engine import DecisionEngine
from test_engine_client import FakeBackend


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/recipes.py"
spec = importlib.util.spec_from_file_location("gemmadecision_documented_recipes", EXAMPLE)
recipes = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recipes
spec.loader.exec_module(recipes)


def engine_preferring(description: str) -> DecisionEngine:
    return DecisionEngine(FakeBackend(scoring=lambda text: 15.0 if text.endswith(description) else -15.0))


def test_list_is_discoverable_without_loading_a_model(monkeypatch, capsys):
    loader = Mock(side_effect=AssertionError("--list must not load model weights"))
    monkeypatch.setattr(DecisionEngine, "from_pretrained", loader)
    assert recipes.main(["--list"]) == 0
    output = capsys.readouterr().out
    assert all(name in output for name in recipes.RECIPES)
    loader.assert_not_called()


def test_running_a_recipe_honors_device_and_loads_once(monkeypatch, capsys):
    engine = engine_preferring(recipes.TEAMS["billing"])
    loader = Mock(return_value=engine)
    monkeypatch.setattr(DecisionEngine, "from_pretrained", loader)
    assert recipes.main(["--recipe", "direct-choice", "--device", "cpu"]) == 0
    assert json.loads(capsys.readouterr().out)["team"] == "billing"
    loader.assert_called_once_with(device="cpu")
    assert len(engine.backend.calls) == 1


def test_choice_and_ranking_preserve_application_labels():
    engine = engine_preferring(recipes.TEAMS["account"])
    assert recipes.direct_choice(engine)["team"] == "account"
    ranked = recipes.ranking(engine)
    assert ranked["ranked"][0]["candidate"] == "account"
    assert {row["candidate"] for row in ranked["ranked"]} == set(recipes.TEAMS)
    assert all("score" in row and "probability" in row for row in ranked["ranked"])


def test_multiple_question_recipe_runs_all_three_types_together():
    engine = engine_preferring(recipes.TEAMS["billing"])
    result = recipes.multiple_questions(engine)
    assert result["team"] == "billing"
    assert {answer["type"] for answer in result["answers"].values()} == {"choice", "noul", "score"}
    assert 0 <= result["urgent_probability"] <= 1
    assert 0 <= result["detail_score_0_to_2"] <= 2
    assert len(engine.backend.calls) == 1
    json.dumps(result)


def test_typed_local_agent_uses_injected_engine_without_loading_another(monkeypatch):
    pytest.importorskip("pydantic_ai.models.decision")
    loader = Mock(side_effect=AssertionError("An injected engine must be reused"))
    monkeypatch.setattr(DecisionEngine, "from_pretrained", loader)
    engine = DecisionEngine(FakeBackend())
    result = recipes.typed_agent(engine)
    validated = recipes.Triage.model_validate(result)
    assert validated.team in recipes.TEAMS
    assert isinstance(validated.urgent, bool)
    assert len(engine.backend.calls) == 1
    loader.assert_not_called()


def test_tool_recommendation_is_bounded_and_does_not_execute():
    engine = engine_preferring(recipes.TOOL_OPTIONS["lookup_order"])
    permitted = recipes.tool_recommendation(engine, allowed_tools=frozenset({"lookup_order"}))
    assert permitted["recommendation"] == "lookup_order"
    assert permitted["allowed_by_application"] is True
    assert permitted["executed"] is False
    forbidden = recipes.tool_recommendation(engine, allowed_tools=frozenset({"search_docs"}))
    assert forbidden["model_choice"] == "lookup_order"
    assert forbidden["recommendation"] == "human_review"
    assert forbidden["allowed_by_application"] is False
    assert forbidden["executed"] is False


def test_review_cannot_become_an_executable_tool_even_if_allowlisted():
    engine = engine_preferring(recipes.TOOL_OPTIONS["human_review"])
    result = recipes.tool_recommendation(engine, allowed_tools=frozenset({"human_review", "unavailable_tool"}))
    assert result["recommendation"] == "human_review"
    assert result["allowed_by_application"] is False
    assert result["executed"] is False


def test_evidence_recipe_has_an_explicit_unknown_relation():
    description = "The supplied evidence is insufficient to determine whether the claim is true."
    engine = engine_preferring(description)
    result = recipes.evidence_relation(engine)
    assert result["relation"] == "unknown"
    assert set(result["probabilities"]) == {"supports", "contradicts", "unknown"}
    assert "evidence:" in engine.backend.calls[0][0] and "claim:" in engine.backend.calls[0][0]


def test_explicit_review_fallback_is_a_candidate_and_propagates():
    description = "Ask a person to clarify an ambiguous request or handle a request outside billing and technical support."
    engine = engine_preferring(description)
    result = recipes.explicit_fallback(engine)
    assert result["destination"] == "human_review"
    assert result["needs_review"] is True
    assert set(result["probabilities"]) == {"billing", "technical", "human_review"}
    other = recipes.explicit_fallback(engine_preferring(recipes.TEAMS["billing"]))
    assert other["destination"] == "billing" and other["needs_review"] is False


def test_application_cutoff_can_send_a_flat_ranking_to_review():
    # This arbitrary test cutoff only exercises the branch; it is not a
    # recommended production threshold or a claim of model calibration.
    engine = DecisionEngine(FakeBackend(scoring=lambda text: 0.0))
    result = recipes.explicit_fallback(engine, minimum_probability=.8)
    assert result["model_choice"] == "billing"
    assert result["destination"] == "human_review"
    assert result["below_application_cutoff"] is True
    no_cutoff = recipes.explicit_fallback(engine)
    assert no_cutoff["destination"] == "billing"
    assert no_cutoff["below_application_cutoff"] is False


def test_invalid_cutoff_fails_before_inference():
    engine = DecisionEngine(FakeBackend())
    with pytest.raises(ValueError, match="between zero and one"):
        recipes.explicit_fallback(engine, minimum_probability=1.2)
    assert engine.backend.calls == []
