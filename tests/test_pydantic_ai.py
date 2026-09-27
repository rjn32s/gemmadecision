"""Exercise the real PydanticAI agent loop without an LLM or model download."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from enum import Enum, IntEnum
from types import SimpleNamespace
from typing import Literal

import httpx
import pytest
from pydantic import BaseModel, Field

pytest.importorskip("pydantic_ai.models.decision")
from pydantic_ai import Agent, UseEnumMemberDocstrings
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.models.decision import (
    ChoiceQuestion,
    DecisionRequest,
    NoulCriteria,
    NoulQuestion,
    ScoreQuestion,
)

from gemmadecision.integrations.pydantic_ai import GemmaDecisionModel


class FakeClient:
    base_url = "http://127.0.0.1:8700"

    def __init__(self, *, fail: Exception | None = None, bad_answers=None):
        self.calls = []
        self.closed = False
        self.fail = fail
        self.bad_answers = bad_answers

    async def system_one(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        answers = {}
        for name, question in kwargs["questions"].items():
            kind = question["type"]
            if kind == "noul":
                answers[name] = {
                    "type": "noul", "noul": 0.8, "probabilities": {"false": 0.2, "true": 0.8}
                }
            elif kind == "score":
                assert len(question["criteria"]) == 3
                answers[name] = {
                    "type": "score", "score": 1.7,
                    "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
                }
            else:
                labels = list(question["criteria"])
                selected = "billing" if "billing" in labels else labels[0]
                probabilities = {label: 0.2 / (len(labels) - 1) for label in labels}
                probabilities[selected] = 0.8
                answers[name] = {
                    "type": "choice", "choice": selected, "probabilities": probabilities
                }
        return SimpleNamespace(
            answers=self.bad_answers if self.bad_answers is not None else answers,
            model="rajan2k/GemmaDecision-270M",
            usage={"input_tokens": 23, "output_tokens": 0},
        )

    async def aclose(self):
        self.closed = True


def test_native_literal_agent_preserves_typed_result_and_probability_provenance():
    client = FakeClient()
    model = GemmaDecisionModel(client=client)
    agent = Agent(
        model,
        output_type=Literal["billing", "account", "technical"],
        instructions="Which team should handle this ticket?",
    )
    result = agent.run_sync("I was charged twice.")
    assert result.output == "billing"
    assert len(client.calls) == 1
    assert "I was charged twice." in json.dumps(client.calls[0]["state"])
    question = next(iter(client.calls[0]["questions"].values()))
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"billing", "account", "technical"}
    assert "Which team should handle this ticket?" in json.dumps(question["instructions"])
    assert result.response.provider_details["probabilities_source"] == "softmax_scores"
    assert result.usage.input_tokens == 23
    assert result.usage.output_tokens == 0


class Department(str, Enum):
    billing = "billing"
    support = "support"


class Ticket(BaseModel):
    department: Department = Field(description="Which team handles this ticket?")
    urgent: bool = Field(description="Does it need attention today?")


def test_multiple_fields_are_one_request_and_boolean_threshold_is_honored():
    client = FakeClient()
    agent = Agent(GemmaDecisionModel(client=client), output_type=Ticket)
    result = agent.run_sync(
        "You charged me twice. Please fix this today.",
        model_settings={"decision_boolean_threshold": 0.9},
    )
    assert result.output == Ticket(department=Department.billing, urgent=False)
    assert len(client.calls) == 1
    assert {q["type"] for q in client.calls[0]["questions"].values()} == {"choice", "noul"}


def test_decisions_protocol_supports_json_wording_noul_criteria_and_score():
    client = FakeClient()
    request = DecisionRequest(
        state={"ticket": "I cannot log in", "attempts": 2},
        questions={
            "priority": ScoreQuestion(criteria=["low", "medium", "high"], instructions={"goal": "Prioritize"}),
            "urgent": NoulQuestion(instructions="Urgent?", criteria=NoulCriteria(true="Needs help today")),
        },
    )
    response = asyncio.run(GemmaDecisionModel(client=client).decide(request, {}))
    assert response.answers["priority"].score == 1.7
    assert response.answers["priority"].probabilities == {0: 0.1, 1: 0.1, 2: 0.8}
    assert response.answers["urgent"].noul == 0.8
    assert client.calls[0]["questions"]["priority"]["instructions"] == {"goal": "Prioritize"}
    assert client.calls[0]["questions"]["urgent"]["criteria"] == {"true": "Needs help today"}


class Impact(UseEnumMemberDocstrings, IntEnum):
    low = 0
    """The customer can complete their task."""
    medium = 1
    """There is a problem but a workaround is available."""
    high = 2
    """The customer cannot complete their task."""


def test_native_described_int_enum_uses_rubric_and_expected_score():
    client = FakeClient()
    agent = Agent(
        GemmaDecisionModel(client=client), output_type=Impact,
        instructions="How severe is this problem?",
    )
    result = agent.run_sync("I cannot log in and cannot use the account at all.")
    assert result.output is Impact.high
    question = next(iter(client.calls[0]["questions"].values()))
    assert question["type"] == "score"
    assert len(question["criteria"]) == 3


def test_request_settings_are_forwarded_without_extra_body_overrides():
    client = FakeClient()
    model = GemmaDecisionModel(client=client)
    request = DecisionRequest(state="test", questions={"ok": NoulQuestion(instructions="Valid?")})
    asyncio.run(model.decide(request, {"timeout": 2, "extra_headers": {"X-Request-ID": "test"}}))
    assert client.calls[0]["timeout"] == 2
    assert client.calls[0]["extra_headers"] == {"X-Request-ID": "test"}
    with pytest.raises(UserError, match="extra_body"):
        asyncio.run(model.decide(request, {"extra_body": {"state": "replacement"}}))
    assert len(client.calls) == 1


def test_unbounded_generated_text_is_rejected_before_any_http_call():
    client = FakeClient()
    class Summary(BaseModel):
        text: str
    with pytest.raises(UserError):
        Agent(GemmaDecisionModel(client=client), output_type=Summary).run_sync("Please summarize this.")
    assert client.calls == []


def test_option_limit_is_rejected_by_native_schema_before_http():
    client = FakeClient()
    enum_type = Enum("TooMany", {f"value_{i}": f"value_{i}" for i in range(65)}, type=str)
    with pytest.raises(UserError):
        Agent(GemmaDecisionModel(client=client), output_type=enum_type).run_sync("Choose something.")
    assert client.calls == []


@pytest.mark.parametrize("bad_answer", [
    {"type": "choice", "choice": "invented", "probabilities": {"a": 0.5, "b": 0.5}},
    {"type": "choice", "choice": "a", "probabilities": {"a": 0.8, "b": 0.8}},
    {"type": "choice", "choice": "a", "probabilities": {"a": float("nan"), "b": 0.5}},
    {"type": "choice", "choice": "a", "probabilities": {"a": 0.3, "b": 0.7}},
    {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}},
    {"type": "noul", "noul": 0.8, "probabilities": {"true": 0.8, "false": 0.2}},
])
def test_invalid_server_answers_cannot_silently_become_agent_outputs(bad_answer):
    model = GemmaDecisionModel(client=FakeClient(bad_answers={"selection": bad_answer}))
    request = DecisionRequest(
        state="test", questions={"selection": ChoiceQuestion(criteria={"a": "First", "b": "Second"})}
    )
    with pytest.raises(UnexpectedModelBehavior):
        asyncio.run(model.decide(request, {}))


def test_server_error_preserves_status_for_pydantic_ai_fallbacks():
    request = httpx.Request("POST", "http://127.0.0.1:8700/v1/system_one")
    response = httpx.Response(503, json={"error": "queue_full"}, request=request)
    failure = httpx.HTTPStatusError("queue full", request=request, response=response)
    model = GemmaDecisionModel(client=FakeClient(fail=failure))
    with pytest.raises(ModelHTTPError) as caught:
        asyncio.run(model.decide(DecisionRequest(state="test", questions={"ok": NoulQuestion(instructions="Valid?")}), {}))
    assert caught.value.status_code == 503
    assert "queue_full" in caught.value.body


def test_async_streaming_returns_a_complete_typed_answer():
    async def run():
        agent = Agent(
            GemmaDecisionModel(client=FakeClient()), output_type=bool, instructions="Is this urgent?"
        )
        async with agent.run_stream("I need this fixed today.") as result:
            return await result.get_output()
    assert asyncio.run(run()) is True


def test_native_output_function_can_route_without_any_generative_model():
    routed = []
    def route(department: Literal["billing", "support"]) -> str:
        """Choose the team to handle a ticket.

        Args:
            department: Which team is responsible for the ticket?
        """
        routed.append(department)
        return f"Send to {department}"

    result = Agent(GemmaDecisionModel(client=FakeClient()), output_type=route).run_sync("Charged twice")
    assert result.output == "Send to billing"
    assert routed == ["billing"]


def test_model_context_closes_owned_client_only(monkeypatch):
    from gemmadecision.integrations import pydantic_ai as integration
    owned, injected = FakeClient(), FakeClient()
    monkeypatch.setattr(integration, "AsyncDecisionClient", lambda **_: owned)

    async def run():
        async with GemmaDecisionModel():
            pass
        async with GemmaDecisionModel(client=injected):
            pass
    asyncio.run(run())
    assert owned.closed is True
    assert injected.closed is False


def test_native_agent_through_real_sdk_http_transport():
    from gemmadecision import AsyncDecisionClient
    calls = []

    def handle(request):
        calls.append(request)
        body = json.loads(request.content)
        assert request.url.path == "/v1/systemone"
        answers = {
            name: {
                "type": "choice", "choice": "billing", "confidence": 0.8,
                "probabilities": {"billing": 0.8, "account": 0.2},
            }
            for name in body["questions"]
        }
        return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": 37}})

    async def run():
        async with AsyncDecisionClient(transport=httpx.MockTransport(handle)) as client:
            model = GemmaDecisionModel(client=client)
            agent = Agent(model, output_type=Literal["billing", "account"])
            return await agent.run("I was charged twice.")

    result = asyncio.run(run())
    assert result.output == "billing"
    assert result.usage.input_tokens == 37
    assert len(calls) == 1


def test_question_limit_prevents_oversized_fanout_before_http():
    client = FakeClient()
    request = DecisionRequest(
        state="test", questions={str(i): NoulQuestion(instructions="Valid?") for i in range(33)}
    )
    with pytest.raises(UserError, match="1 to 32"):
        asyncio.run(GemmaDecisionModel(client=client).decide(request, {}))
    assert client.calls == []


class LocalBackend:
    max_state_tokens = 2048
    max_action_tokens = 768
    context_limit = 32768

    def __init__(self):
        self.calls = []
        self.threads = []

    def count_tokens(self, text):
        return len(text)

    def score_pairs(self, texts):
        self.calls.append(texts)
        self.threads.append(threading.get_ident())
        return [2.0 if text.endswith("billing") else -2.0 for text in texts]


def test_local_constructor_is_lazy_and_shares_engine_with_simple_entry_points(monkeypatch):
    import gemmadecision
    from gemmadecision import simple
    from gemmadecision.engine import DecisionEngine
    backend = LocalBackend()
    engine = DecisionEngine(backend)
    loaded = []
    monkeypatch.setattr(simple, "_engine", None)

    def load(*args, **kwargs):
        loaded.append(threading.get_ident())
        return engine
    monkeypatch.setattr(DecisionEngine, "from_pretrained", load)
    first, second = GemmaDecisionModel.local(), GemmaDecisionModel.local()
    assert loaded == []
    assert backend.calls == []
    assert first.base_url == "local://gemmadecision"
    for model in (first, second):
        agent = Agent(model, output_type=Literal["billing", "account"])
        assert agent.run_sync("I was charged twice.").output == "billing"
    assert len(loaded) == 1
    assert loaded[0] != threading.get_ident()
    assert all(identity != threading.get_ident() for identity in backend.threads)
    assert gemmadecision.decide("Another ticket", ["billing", "account"]) == "billing"
    assert len(loaded) == 1
    assert len(backend.calls) == 3


def test_local_explicit_engine_reuses_weights_after_model_context_closes(monkeypatch):
    from gemmadecision import simple
    from gemmadecision.engine import DecisionEngine
    backend = LocalBackend()
    engine = DecisionEngine(backend)
    monkeypatch.setattr(simple, "get_engine", lambda: pytest.fail("An injected engine must bypass default loading"))

    async def run():
        async with GemmaDecisionModel.local(engine=engine) as model:
            agent = Agent(model, output_type=Literal["billing", "account"])
            one = await agent.run("First ticket")
        two = await Agent(model, output_type=Literal["billing", "account"]).run("Second ticket")
        return one, two
    one, two = asyncio.run(run())
    assert one.output == two.output == "billing"
    assert len(backend.calls) == 2


def test_local_input_errors_are_user_errors_and_invalid_scores_are_model_errors():
    from gemmadecision.engine import DecisionEngine
    backend = LocalBackend()
    backend.max_state_tokens = 4
    model = GemmaDecisionModel.local(engine=DecisionEngine(backend))
    with pytest.raises(UserError, match="limit"):
        Agent(model, output_type=Literal["billing", "account"]).run_sync("A long ticket")
    assert backend.calls == []
    backend.max_state_tokens = 2048
    backend.score_pairs = lambda texts: [float("nan") for _ in texts]
    with pytest.raises(ModelAPIError, match="invalid score"):
        Agent(model, output_type=Literal["billing", "account"]).run_sync("Ticket")


def test_local_headers_and_invalid_timeout_fail_before_inference():
    from gemmadecision.engine import DecisionEngine
    backend = LocalBackend()
    agent = Agent(GemmaDecisionModel.local(engine=DecisionEngine(backend)), output_type=Literal["billing", "account"])
    with pytest.raises(UserError, match="extra_headers"):
        agent.run_sync("Ticket", model_settings={"extra_headers": {"X-Extra": "unused"}})
    with pytest.raises(UserError, match="positive finite"):
        agent.run_sync("Ticket", model_settings={"timeout": -1})
    assert backend.calls == []


def test_local_timeout_during_loading_skips_later_inference(monkeypatch):
    from gemmadecision import simple
    from gemmadecision.engine import DecisionEngine
    backend = LocalBackend()
    engine = DecisionEngine(backend)
    def slow_load():
        time.sleep(0.1)
        return engine
    monkeypatch.setattr(simple, "get_engine", slow_load)

    async def run():
        agent = Agent(GemmaDecisionModel.local(), output_type=Literal["billing", "account"])
        with pytest.raises(ModelAPIError, match="timed out"):
            await agent.run("Ticket", model_settings={"timeout": 0.02})
        await asyncio.sleep(0.12)
    asyncio.run(run())
    assert backend.calls == []


def test_concurrent_local_agents_serialize_shared_backend_access():
    from gemmadecision.engine import DecisionEngine
    class ConcurrentBackend(LocalBackend):
        active = 0
        peak = 0
        def score_pairs(self, texts):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                time.sleep(0.02)
                return super().score_pairs(texts)
            finally:
                self.active -= 1

    backend = ConcurrentBackend()
    model = GemmaDecisionModel.local(engine=DecisionEngine(backend))
    agent = Agent(model, output_type=Literal["billing", "account"])
    async def run():
        return await asyncio.gather(agent.run("First ticket"), agent.run("Second ticket"))
    results = asyncio.run(run())
    assert [result.output for result in results] == ["billing", "billing"]
    assert len(backend.calls) == 2
    assert backend.peak == 1
