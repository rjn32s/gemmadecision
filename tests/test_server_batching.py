"""HTTP and scheduling contracts using deterministic, weight-free backends."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import threading
import time

from asgi_lifespan import LifespanManager
import httpx
import pytest

from gemmadecision.batching import MicroBatcher, QueueFullError
from gemmadecision.engine import DecisionEngine
from gemmadecision.server import create_app
from gemmadecision.types import Choice, SystemOneRequest


class Backend:
    max_state_tokens = 2048
    max_action_tokens = 768
    context_limit = 32768

    def __init__(self):
        self.calls = []
        self.bad_scores = False

    def count_tokens(self, text):
        return len(text)

    def score_pairs(self, texts):
        self.calls.append(list(texts))
        if self.bad_scores:
            return [float("nan") for _ in texts]
        return [2.0 if text.endswith("B") else -2.0 for text in texts]


class BlockingBackend(Backend):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def score_pairs(self, texts):
        self.entered.set()
        if not self.release.wait(timeout=3):
            raise RuntimeError("Test failed to release its fake inference backend")
        return super().score_pairs(texts)


def request(state="State"):
    return SystemOneRequest(state=state, questions={"q": Choice(instructions="Pick", criteria={"a": "A", "b": "B"})})


def body(state="State"):
    return request(state).model_dump(mode="json")


@asynccontextmanager
async def serving(engine, *, config=None, api_key=None):
    app = create_app(engine=engine, config=config or {}, api_key=api_key)
    async with LifespanManager(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield app, client


@pytest.mark.asyncio
async def test_microbatch_combines_requests_once_and_restores_each_answer_order():
    backend = Backend()
    engine = DecisionEngine(backend)
    plans = [engine.prepare(request(state)) for state in ("one", "two", "three")]
    batcher = MicroBatcher(engine, batch_requests=3, wait_ms=20)
    await batcher.start()
    try:
        results = await asyncio.gather(*(batcher.submit(plan) for plan in plans))
    finally:
        await batcher.close()
    assert len(backend.calls) == 1
    assert backend.calls[0] == [text for plan in plans for text in plan.texts]
    answers = [engine.finish(plan, scored).answers["q"] for plan, scored in zip(plans, results)]
    assert [answer.choice for answer in answers] == ["b", "b", "b"]
    assert all(answer.scores == {"a": -2.0, "b": 2.0} for answer in answers)
    assert batcher.completed_requests == 3
    assert batcher.model_batches == 1


@pytest.mark.asyncio
async def test_full_queue_fails_promptly_while_another_request_is_computing():
    backend = BlockingBackend()
    engine = DecisionEngine(backend)
    plan = engine.prepare(request())
    batcher = MicroBatcher(engine, queue_size=1, batch_requests=1, wait_ms=0)
    await batcher.start()
    first = asyncio.create_task(batcher.submit(plan))
    second = None
    try:
        assert await asyncio.to_thread(backend.entered.wait, 1)
        second = asyncio.create_task(batcher.submit(plan))
        await asyncio.sleep(0)
        started = time.perf_counter()
        with pytest.raises(QueueFullError):
            await batcher.submit(plan)
        assert time.perf_counter() - started < 0.2
        backend.release.set()
        await asyncio.gather(first, second)
    finally:
        backend.release.set()
        await batcher.close()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_timeout_returns_without_blocking_event_loop_or_retrying_compute():
    backend = BlockingBackend()
    engine = DecisionEngine(backend)
    plan = engine.prepare(request())
    batcher = MicroBatcher(engine, wait_ms=0, timeout_s=0.03)
    await batcher.start()
    task = asyncio.create_task(batcher.submit(plan))
    try:
        assert await asyncio.to_thread(backend.entered.wait, 1)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=0.3)
        assert not backend.release.is_set()
        assert batcher.completed_requests == 0
    finally:
        backend.release.set()
        await asyncio.sleep(0.02)
        await batcher.close()
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_cancelled_queued_request_is_skipped_without_an_extra_backend_call():
    backend = BlockingBackend()
    engine = DecisionEngine(backend)
    first_plan = engine.prepare(request("first"))
    second_plan = engine.prepare(request("cancelled"))
    batcher = MicroBatcher(engine, batch_requests=1, wait_ms=0)
    await batcher.start()
    first = asyncio.create_task(batcher.submit(first_plan))
    second = None
    try:
        assert await asyncio.to_thread(backend.entered.wait, 1)
        second = asyncio.create_task(batcher.submit(second_plan))
        await asyncio.sleep(0)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        backend.release.set()
        await first
        await asyncio.wait_for(batcher.queue.join(), timeout=0.5)
    finally:
        backend.release.set()
        await batcher.close()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
    assert len(backend.calls) == 1
    assert all("cancelled" not in text for text in backend.calls[0])


@pytest.mark.asyncio
async def test_health_and_models_work_without_inference():
    backend = Backend()
    async with serving(DecisionEngine(backend)) as (app, client):
        for endpoint in ("/health", "/ready"):
            response = await client.get(endpoint)
            assert response.status_code == 200
            assert response.json()["status"] == "ready"
            assert response.json()["completed_requests"] == 0
        models = (await client.get("/v1/models")).json()["models"]
        assert models[0]["task"] == "typed-decisions"
    assert backend.calls == []


@pytest.mark.asyncio
async def test_api_authentication_rejects_missing_and_wrong_keys_before_compute():
    backend = Backend()
    async with serving(DecisionEngine(backend), api_key="server-key") as (_, client):
        for headers in ({}, {"Authorization": "Bearer wrong"}, {b"Authorization": b"Bearer \xff"}):
            response = await client.post("/v1/systemone", json=body(), headers=headers)
            assert response.status_code == 401
        assert backend.calls == []
        accepted = await client.post("/v1/systemone", json=body(), headers={"Authorization": "Bearer server-key"})
        assert accepted.status_code == 200
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_http_question_types_and_rank_keep_the_same_decision_contract():
    backend = Backend()
    async with serving(DecisionEngine(backend)) as (_, client):
        response = await client.post("/v1/systemone", json={
            "state": {"ticket": "Help me"},
            "questions": {
                "choice": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}},
                "noul": {"type": "noul", "instructions": "Valid?", "criteria": {"no": "A", "yes": "B"}},
                "score": {"type": "score", "instructions": "Rate", "criteria": ["A", "B"]},
            },
        })
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["answers"]["choice"]["choice"] == "b"
        assert payload["answers"]["noul"]["noul"] > 0.5
        assert payload["answers"]["score"]["score"] > 0.5
        assert payload["usage"]["output_tokens"] == 0
        assert payload["probabilities_source"] == "softmax_ranking_scores_external_temperature"
        ranked = await client.post("/v1/rank", json={"state": "Rank these", "candidates": {"first": "A", "second": "B"}})
        assert ranked.status_code == 200
        assert [item["candidate"] for item in ranked.json()["ranked"]] == ["second", "first"]
        assert [item["rank"] for item in ranked.json()["ranked"]] == [1, 2]


@pytest.mark.asyncio
async def test_body_limit_rejects_large_and_chunked_payloads_before_compute():
    backend = Backend()
    async with serving(DecisionEngine(backend)) as (_, client):
        response = await client.post("/v1/systemone", content=b"x" * (2 * 1024 * 1024 + 1))
        assert response.status_code == 413
        async def chunks():
            yield b"x" * (1024 * 1024)
            yield b"x" * (1024 * 1024 + 1)
        chunked = await client.post("/v1/systemone", content=chunks())
        assert chunked.status_code == 413
    assert backend.calls == []


@pytest.mark.asyncio
async def test_context_refusal_is_422_without_compute_or_truncation():
    backend = Backend()
    backend.max_state_tokens = 10
    async with serving(DecisionEngine(backend)) as (_, client):
        response = await client.post("/v1/systemone", json=body("x" * 11))
        assert response.status_code == 422
        assert "limit" in response.json()["detail"]
    assert backend.calls == []


@pytest.mark.asyncio
async def test_invalid_preflight_does_not_poison_a_concurrent_valid_request():
    backend = Backend()
    backend.max_state_tokens = 20
    async with serving(DecisionEngine(backend), config={"batch_wait_ms": 20}) as (_, client):
        valid, invalid = await asyncio.gather(
            client.post("/v1/systemone", json=body("OK")),
            client.post("/v1/systemone", json=body("x" * 21)),
        )
        assert valid.status_code == 200
        assert valid.json()["answers"]["q"]["choice"] == "b"
        assert invalid.status_code == 422
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 2
    assert all("xxxxxxxx" not in text for text in backend.calls[0])


@pytest.mark.asyncio
async def test_http_backpressure_covers_requests_waiting_for_preparation():
    backend = BlockingBackend()
    config = {"queue_size": 1, "batch_requests": 1, "batch_wait_ms": 0}
    async with serving(DecisionEngine(backend), config=config) as (app, client):
        first = asyncio.create_task(client.post("/v1/systemone", json=body("first")))
        second = None
        try:
            assert await asyncio.to_thread(backend.entered.wait, 1)
            second = asyncio.create_task(client.post("/v1/systemone", json=body("second")))
            async def queued():
                while app.state.batcher.queue.qsize() != 1:
                    await asyncio.sleep(0)
            await asyncio.wait_for(queued(), 0.3)
            rejected = await asyncio.wait_for(client.post("/v1/systemone", json=body("third")), 0.3)
            assert rejected.status_code == 503
            assert rejected.headers["retry-after"] == "1"
            backend.release.set()
            responses = await asyncio.gather(first, second)
            assert [response.status_code for response in responses] == [200, 200]
        finally:
            backend.release.set()
            await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_invalid_scores_return_500_and_do_not_poison_the_cache():
    backend = Backend()
    backend.bad_scores = True
    async with serving(DecisionEngine(backend, cache_size=8)) as (_, client):
        failed = await client.post("/v1/systemone", json=body())
        assert failed.status_code == 500
        assert failed.json() == {"detail": "Inference backend failed"}
        backend.bad_scores = False
        recovered = await client.post("/v1/systemone", json=body())
        assert recovered.status_code == 200
        assert recovered.json()["answers"]["q"]["choice"] == "b"
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_http_timeout_returns_504_while_health_stays_responsive():
    backend = BlockingBackend()
    async with serving(DecisionEngine(backend), config={"request_timeout": 0.03, "batch_wait_ms": 0}) as (_, client):
        pending = asyncio.create_task(client.post("/v1/systemone", json=body()))
        try:
            assert await asyncio.to_thread(backend.entered.wait, 1)
            response, health = await asyncio.gather(
                asyncio.wait_for(pending, 0.3), asyncio.wait_for(client.get("/health"), 0.3)
            )
            assert response.status_code == 504
            assert health.status_code == 200
        finally:
            backend.release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await asyncio.sleep(0.02)
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_request_timeout_also_bounds_preparation_before_the_inference_queue():
    class SlowPrepareEngine(DecisionEngine):
        def prepare(self, request):
            time.sleep(0.15)
            return super().prepare(request)

    backend = Backend()
    async with serving(SlowPrepareEngine(backend), config={"request_timeout": 0.03}) as (_, client):
        started = time.perf_counter()
        response = await client.post("/v1/systemone", json=body())
        elapsed = time.perf_counter() - started
        assert response.status_code == 504
        assert elapsed < 0.12
        await asyncio.sleep(0.16)
    assert backend.calls == []
