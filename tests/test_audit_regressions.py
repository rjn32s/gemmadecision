"""Release-audit regressions using a bounded, weight-free inference backend."""
import asyncio

from asgi_lifespan import LifespanManager
import httpx
import pytest

from gemmadecision.backends.torch import plan_batches
from gemmadecision.engine import DecisionEngine
from gemmadecision.server import create_app


class SmallBudgetBackend:
    max_state_tokens = 2048
    max_action_tokens = 768
    context_limit = 32768
    max_batch_tokens = 64

    def __init__(self):
        self.calls = []

    def count_tokens(self, text):
        return len(text)

    def score_pairs(self, texts):
        self.calls.append(list(texts))
        # The same hard per-input bound enforced by the real Torch scheduler.
        plan_batches([len(text) for text in texts], self.max_batch_tokens, 32)
        return [1.0 if text.endswith("B") else 0.0 for text in texts]


def payload(state, rank):
    if rank:
        return {"state": state, "question": "Pick", "candidates": {"a": "A", "b": "B"}}
    return {"state": state, "questions": {
        "q": {"type": "choice", "instructions": "Pick", "criteria": {"a": "A", "b": "B"}},
    }}


@pytest.mark.asyncio
@pytest.mark.parametrize("rank", [False, True], ids=["system-one", "rank"])
async def test_custom_token_budget_refusal_does_not_poison_coalesced_valid_request(rank):
    backend = SmallBudgetBackend()
    app = create_app(engine=DecisionEngine(backend), config={"batch_wait_ms": 30, "batch_requests": 2})
    route = "/v1/rank" if rank else "/v1/systemone"
    async with LifespanManager(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            valid, oversized = await asyncio.gather(
                client.post(route, json=payload("Short state", rank)),
                client.post(route, json=payload("x" * 80, rank)),
            )
    assert valid.status_code == 200, valid.text
    assert oversized.status_code == 422, oversized.text
    if rank:
        assert valid.json()["ranked"][0]["candidate"] == "b"
    else:
        assert valid.json()["answers"]["q"]["choice"] == "b"
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 2
    assert all(len(text) <= backend.max_batch_tokens for text in backend.calls[0])


def test_direct_rank_observes_backend_token_budget_before_inference():
    backend = SmallBudgetBackend()
    engine = DecisionEngine(backend)
    with pytest.raises(ValueError):
        engine.rank("x" * 80, {"a": "A", "b": "B"}, question="Pick")
    assert backend.calls == []
    result = engine.rank("Short state", {"a": "A", "b": "B"}, question="Pick")
    assert result.ranked[0].candidate == "b"
    assert len(backend.calls) == 1
