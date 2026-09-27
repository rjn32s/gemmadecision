"""Decision protocol and SDK contracts, without model weights or an ML runtime."""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest
from pydantic import ValidationError

from gemmadecision import AsyncDecisionClient, Choice, DecisionClient, Noul, Score
from gemmadecision.constants import MODEL_ID, TEMPERATURE
from gemmadecision.engine import DecisionEngine
from gemmadecision.types import RankRequest, SystemOneRequest


class FakeBackend:
    """Character counts make exact limit boundaries visible without a tokenizer."""

    max_state_tokens = 2048
    max_action_tokens = 768
    context_limit = 32768

    def __init__(self, scoring=None):
        self.calls = []
        self.scoring = scoring or (lambda text: float(sum(text.encode("utf-8")) % 31))

    def count_tokens(self, text):
        return len(text)

    def score_pairs(self, texts):
        self.calls.append(list(texts))
        return [self.scoring(text) for text in texts]


def choose(instructions="Which option?", criteria=None):
    return Choice(instructions=instructions, criteria=criteria or {"a": "Option A", "b": "Option B"})


def test_structured_input_uses_frozen_v4_text_layout_and_json_string_equivalence():
    # Golden text from the published v0.4.0 common.schema_state + CLM to_text
    # contract, including nested indentation, lowercase booleans and key order.
    state = {"customer": "Ana", "details": {"active": True, "items": ["a", "b"]}}
    question = Choice(
        instructions={"goal": "Route this request", "when": ["today"]},
        criteria={"first": {"team": "Billing", "available": True}, "second": None},
    )
    engine = DecisionEngine(FakeBackend())
    plan = engine.prepare(SystemOneRequest(state=state, questions={"route": question}))
    rendered = (
        "customer: Ana\n\ndetails:\n  active: true\n  items:\n    - a\n    - b"
        "\n\ngoal: Route this request\n\nwhen:\n  - today"
    )
    assert plan.texts == [
        rendered + "\n\nCandidate action:\nteam: Billing\n\navailable: true",
        rendered + "\n\nCandidate action:\nsecond",
    ]
    encoded_state = json.dumps(state, ensure_ascii=False)
    other = engine.prepare(SystemOneRequest(state=encoded_state, questions={"route": question}))
    assert other.texts == plan.texts
    assert engine.backend.calls == []


@pytest.mark.parametrize("criteria", [
    {"false": "It can wait", "true": "Needs attention"},
    {"no": "It can wait", "yes": "Needs attention"},
    {"false": "It can wait", "yes": "Needs attention"},
    {"no": "It can wait", "true": "Needs attention"},
])
def test_noul_aliases_preserve_native_false_true_semantics(criteria):
    backend = FakeBackend(scoring=lambda text: 2.0 if "true: Needs attention" in text else -2.0)
    engine = DecisionEngine(backend)
    response = engine.system_one("Customer request", {"urgent": Noul(instructions="Urgent?", criteria=criteria)})
    answer = response.answers["urgent"]
    assert backend.calls == [[
        "Customer request\n\nUrgent?\n\nCandidate action:\nfalse: It can wait",
        "Customer request\n\nUrgent?\n\nCandidate action:\ntrue: Needs attention",
    ]]
    assert set(answer.probabilities) == {"false", "true"}
    assert answer.noul == answer.probabilities["true"]
    assert answer.noul > 0.5


def test_default_noul_candidates_match_frozen_boolean_prompt():
    backend = FakeBackend()
    DecisionEngine(backend).system_one("Please help", {"q": Noul(instructions="Urgent?")})
    assert backend.calls[0] == [
        "Please help\n\nUrgent?\n\nCandidate action:\nfalse: No. This is false: Urgent?",
        "Please help\n\nUrgent?\n\nCandidate action:\ntrue: Yes. This is true: Urgent?",
    ]


@pytest.mark.parametrize("criteria", [
    {"false": "no", "no": "no", "true": "yes"},
    {"false": "no", "true": "yes", "yes": "yes"},
    {"accept": "yes", "reject": "no"},
])
def test_ambiguous_or_unknown_boolean_keys_fail_before_compute(criteria):
    backend = FakeBackend()
    with pytest.raises(ValueError):
        DecisionEngine(backend).system_one("Request", {"q": Noul(instructions="Valid?", criteria=criteria)})
    assert backend.calls == []


def test_score_is_distribution_expectation_and_choice_preserves_labels():
    backend = FakeBackend(scoring=lambda text: {"low": 0.0, "medium": 1.0, "high": 2.0}[text.rsplit("\n", 1)[-1]])
    response = DecisionEngine(backend).system_one("Ticket", {
        "severity": Score(instructions="How severe?", criteria=["low", "medium", "high"]),
        "route": Choice(instructions="What priority?", criteria={"l": "low", "m": "medium", "h": "high"}),
    })
    weights = [math.exp(score / TEMPERATURE) for score in [0.0, 1.0, 2.0]]
    expected = [value / sum(weights) for value in weights]
    assert response.answers["severity"].probabilities == pytest.approx(dict(zip(["0", "1", "2"], expected)))
    assert response.answers["severity"].score == pytest.approx(expected[1] + 2 * expected[2])
    assert response.answers["route"].choice == "h"
    assert set(response.answers["route"].probabilities) == {"l", "m", "h"}
    assert response.usage.output_tokens == 0


@pytest.mark.parametrize("limit_kind", ["state", "action", "joint"])
def test_input_limits_reject_before_compute_without_silent_truncation(limit_kind):
    backend = FakeBackend()
    if limit_kind == "state":
        backend.max_state_tokens = 4
        state, question = "12345", choose(instructions="")
    elif limit_kind == "action":
        backend.max_action_tokens = 3
        state, question = "x", choose(criteria={"a": "1234", "b": "123"})
    else:
        backend.context_limit = 12
        state, question = "x", choose(instructions="", criteria={"a": "A", "b": "B"})
    with pytest.raises(ValueError, match="limit"):
        DecisionEngine(backend).system_one(state, {"q": question})
    assert backend.calls == []


def test_state_and_action_limits_allow_the_exact_boundary():
    backend = FakeBackend()
    backend.max_state_tokens, backend.max_action_tokens = 4, 3
    DecisionEngine(backend).system_one("1234", {"q": choose(instructions="", criteria={"a": "123", "b": "abc"})})
    assert len(backend.calls) == 1


@pytest.mark.parametrize("state,instructions", [("", ""), ("   ", "\n"), (None, None)])
def test_empty_rendered_state_and_question_are_rejected_like_frozen_v4(state, instructions):
    backend = FakeBackend()
    with pytest.raises(ValueError, match="nonempty"):
        DecisionEngine(backend).system_one(state, {"q": choose(instructions=instructions)})
    assert backend.calls == []


@pytest.mark.parametrize("criteria", [
    {"a": "same", "b": "same"},
    {"a": " ", "b": "Useful description"},
])
def test_invalid_candidate_text_does_not_reach_backend(criteria):
    backend = FakeBackend()
    with pytest.raises(ValueError):
        DecisionEngine(backend).system_one("State", {"q": choose(criteria=criteria)})
    assert backend.calls == []


def test_candidate_and_question_count_limits_are_validated():
    backend = FakeBackend()
    engine = DecisionEngine(backend)
    with pytest.raises(ValidationError):
        engine.system_one("state", {"q": choose(criteria={str(i): str(i) for i in range(65)})})
    with pytest.raises(ValidationError):
        engine.system_one("state", {str(i): choose() for i in range(33)})
    with pytest.raises(ValueError, match="256"):
        engine.system_one("state", {str(i): choose(criteria={str(j): str(j) for j in range(64)}) for i in range(5)})
    assert backend.calls == []


def test_unsupported_model_cannot_silently_use_the_default_weights():
    backend = FakeBackend()
    engine = DecisionEngine(backend)
    with pytest.raises(ValueError, match="Unknown model"):
        engine.prepare(SystemOneRequest(state="State", questions={"q": choose()}, model="somebody/other-model"))
    with pytest.raises(ValueError, match="Unknown model"):
        engine.prepare_rank(RankRequest(state="State", candidates=["A", "B"], model="wrong"))
    assert backend.calls == []


def test_batch_deduplication_and_cache_account_only_actually_encoded_joint_tokens():
    backend = FakeBackend()
    engine = DecisionEngine(backend, cache_size=8)
    question = choose()
    plan_a = engine.prepare(SystemOneRequest(state="Same state", questions={"first": question}))
    plan_b = engine.prepare(SystemOneRequest(state="Same state", questions={"second": question}))
    results = engine.score_many([plan_a, plan_b])
    assert backend.calls == [plan_a.texts]
    assert results[0][0] == results[1][0]
    assert results[0][1].input_tokens == sum(len(text) for text in plan_a.texts)
    assert results[0][1].cached_pairs == 0
    assert results[1][1].input_tokens == 0
    assert results[1][1].cached_pairs == 2
    cached = engine.score_many([plan_a])[0]
    assert cached[1].input_tokens == 0
    assert cached[1].total_tokens == 0
    assert cached[1].cached_pairs == 2
    assert len(backend.calls) == 1


def test_changing_state_or_question_recomputes_candidates_and_can_change_decision():
    def score(text):
        is_b = text.endswith("Option B")
        prefer_b = "prefer-B" in text
        return 2.0 if is_b == prefer_b else -2.0

    backend = FakeBackend(scoring=score)
    engine = DecisionEngine(backend, cache_size=32)
    candidates = {"a": "Option A", "b": "Option B"}
    assert engine.decide("prefer-A", candidates, question="Pick").choice == "a"
    assert engine.decide("prefer-B", candidates, question="Pick").choice == "b"
    assert engine.decide("prefer-A", candidates, question="prefer-B").choice == "b"
    assert len(backend.calls) == 3
    assert all(len(call) == 2 for call in backend.calls)


def test_label_changes_and_permutations_reuse_only_identical_joint_inputs():
    backend = FakeBackend(scoring=lambda text: 2.0 if text.endswith("Option B") else -2.0)
    engine = DecisionEngine(backend, cache_size=8)
    original = engine.decide("State", {"a": "Option A", "b": "Option B"})
    permuted = engine.decide("State", {"renamed_b": "Option B", "renamed_a": "Option A"})
    assert original.choice == "b"
    assert permuted.choice == "renamed_b"
    assert original.probabilities["b"] == permuted.probabilities["renamed_b"]
    assert len(backend.calls) == 1


def test_cache_is_disabled_by_default_and_explicit_cache_can_be_cleared():
    backend = FakeBackend()
    engine = DecisionEngine(backend)
    for _ in range(2):
        engine.decide("Private request", {"a": "A", "b": "B"})
    assert len(backend.calls) == 2
    cached = DecisionEngine(backend, cache_size=2)
    cached.decide("Private request", {"a": "A", "b": "B"})
    cached.decide("Private request", {"a": "A", "b": "B"})
    assert len(backend.calls) == 3
    cached.clear_cache()
    cached.decide("Private request", {"a": "A", "b": "B"})
    assert len(backend.calls) == 4


@pytest.mark.parametrize("bad_scores", [[float("nan"), 0.0], [float("inf"), 0.0], [0.0], []])
def test_invalid_backend_scores_fail_and_are_not_cached(bad_scores):
    backend = FakeBackend()
    calls = []
    def bad(texts):
        calls.append(texts)
        return bad_scores
    backend.score_pairs = bad
    engine = DecisionEngine(backend, cache_size=8)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="invalid score"):
            engine.decide("State", {"a": "A", "b": "B"})
    assert len(calls) == 2


def test_client_import_and_lazy_engine_import_need_no_ml_runtime():
    root = Path(__file__).resolve().parents[1] / "python"
    script = """
import importlib.abc
import sys
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'vllm', 'huggingface_hub', 'pydantic_ai'}:
            raise RuntimeError('Unexpected heavyweight import: ' + fullname)
sys.meta_path.insert(0, BlockML())
import gemmadecision
from gemmadecision import DecisionClient, AsyncDecisionClient, DecisionEngine
print('lightweight')
"""
    environment = {**os.environ, "PYTHONPATH": str(root)}
    result = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "lightweight"


def test_async_client_serializes_questions_and_forwards_timeout_and_headers():
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={
            "model": MODEL_ID,
            "answers": {"q": {"type": "choice", "choice": "b", "confidence": 0.8,
                              "probabilities": {"a": 0.2, "b": 0.8}}},
            "usage": {"input_tokens": 41},
        })

    async def run():
        async with AsyncDecisionClient(
            base_url="http://service:8700/", api_key="test-only", transport=httpx.MockTransport(handle)
        ) as client:
            return await client.system_one(
                state={"request": "Hello"}, questions={"q": choose()},
                timeout=2.5, extra_headers={"X-Request-ID": "request-123"},
            )

    response = asyncio.run(run())
    request = calls[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/systemone"
    assert request.headers["authorization"] == "Bearer test-only"
    assert request.headers["x-request-id"] == "request-123"
    assert request.extensions["timeout"] == {"connect": 2.5, "read": 2.5, "write": 2.5, "pool": 2.5}
    body = json.loads(request.content)
    assert body["state"] == {"request": "Hello"}
    assert body["questions"]["q"]["type"] == "choice"
    assert response.answers["q"].choice == "b"
    assert response.usage.input_tokens == 41


def test_sdk_serialization_preserves_nullable_question_fields_accepted_by_server():
    calls = []
    def handle(request):
        parsed = SystemOneRequest.model_validate(json.loads(request.content))
        calls.append(parsed)
        return httpx.Response(200, json={
            "answers": {"q": {"type": "choice", "choice": "a", "confidence": 0.8,
                              "probabilities": {"a": 0.8, "b": 0.2}}}, "usage": {},
        })
    with DecisionClient(transport=httpx.MockTransport(handle)) as client:
        client.system_one("Classify this state", {"q": Choice(instructions=None, criteria={"a": "First", "b": "Second"})})
    assert calls[0].questions["q"].instructions is None


@pytest.mark.parametrize("status", [307, 401, 422, 429, 503])
def test_clients_do_not_retry_or_follow_redirects(status):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "deliberate"}, headers={"location": "http://elsewhere/"})

    with DecisionClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(httpx.HTTPStatusError) as error:
            client.system_one("State", {"q": choose()})
        assert error.value.response.status_code == status

    async def run():
        async with AsyncDecisionClient(transport=httpx.MockTransport(handle)) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.system_one("State", {"q": choose()})
    asyncio.run(run())
    assert len(calls) == 2


def test_closed_async_client_fails_without_a_network_request():
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={})
    async def run():
        client = AsyncDecisionClient(transport=httpx.MockTransport(handle))
        await client.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await client.system_one("State", {"q": choose()})
    asyncio.run(run())
    assert calls == []
