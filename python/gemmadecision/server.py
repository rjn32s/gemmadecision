"""ASGI application served by Granian's Rust HTTP runtime."""
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import os
import time

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from .batching import MicroBatcher, QueueFullError
from .constants import MODEL_ID
from . import __version__
from .engine import DecisionEngine
from .types import SystemOneRequest, SystemOneResponse, RankRequest, RankingResponse


class BodyLimitMiddleware:
    def __init__(self, app, limit=2 * 1024 * 1024):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST':
            return await self.app(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            if message['type'] != 'http.request':
                continue
            total += len(message.get('body', b''))
            if total > self.limit:
                return await JSONResponse({'detail': 'Request body exceeds 2 MiB'}, status_code=413)(scope, receive, send)
            chunks.append(message.get('body', b''))
            if not message.get('more_body'):
                break
        body = b''.join(chunks)
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            return await receive()
        return await self.app(scope, replay, send)


def create_app(engine=None, *, config=None, api_key=None):
    options = dict(config if config is not None else json.loads(os.getenv('GEMMADECISION_CONFIG', '{}')))
    key = api_key if api_key is not None else os.getenv('GEMMADECISION_API_KEY')

    @asynccontextmanager
    async def lifespan(app):
        instance = engine
        if instance is None:
            kwargs = {k: options[k] for k in ('model_path', 'device', 'backend', 'strict', 'max_batch_tokens',
                       'max_batch_size', 'cache_size', 'offline', 'gpu_memory_utilization') if k in options}
            instance = await asyncio.to_thread(DecisionEngine.from_pretrained, **kwargs)
        app.state.engine = instance
        app.state.batcher = MicroBatcher(instance, queue_size=options.get('queue_size', 128),
                                        batch_requests=options.get('batch_requests', 32),
                                        wait_ms=options.get('batch_wait_ms', 1.0),
                                        timeout_s=options.get('request_timeout', 60))
        await app.state.batcher.start()
        try:
            yield
        finally:
            await app.state.batcher.close()

    app = FastAPI(title='GemmaDecision', version=__version__, lifespan=lifespan,
                  description='Typed local decisions. Probabilities are derived from frozen ranking scores.')
    app.add_middleware(BodyLimitMiddleware)

    async def authorize(request: Request):
        if key and not hmac.compare_digest(request.headers.get('Authorization', '').encode(), ('Bearer ' + key).encode()):
            raise HTTPException(401, 'Missing or invalid API key')

    @app.get('/health')
    @app.get('/ready')
    async def health():
        batcher = getattr(app.state, 'batcher', None)
        if batcher is None or batcher.closed or batcher.task.done():
            raise HTTPException(503, 'Model is not ready')
        return {'status': 'ready', 'serving_runtime': 'granian-rust',
                **app.state.engine.metadata(), 'queued_requests': batcher.queue.qsize(),
                'completed_requests': batcher.completed_requests, 'model_batches': batcher.model_batches}

    @app.get('/v1/models')
    async def models():
        return {'models': [{'id': MODEL_ID, 'name': MODEL_ID, 'task': 'typed-decisions'}]}

    async def infer(request, rank=False):
        started = time.perf_counter()
        engine = app.state.engine
        try:
            plan, scored = await app.state.batcher.submit_request(request, rank=rank)
            elapsed = (time.perf_counter() - started) * 1000
            return engine.finish_rank(plan, scored, latency_ms=elapsed) if rank else engine.finish(plan, scored, latency_ms=elapsed)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except QueueFullError as exc:
            raise HTTPException(503, str(exc), headers={'Retry-After': '1'}) from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(504, 'Decision timed out; no automatic retry was performed') from exc
        except RuntimeError as exc:
            raise HTTPException(500, 'Inference backend failed') from exc

    @app.post('/v1/systemone', response_model=SystemOneResponse, dependencies=[Depends(authorize)])
    async def system_one(request: SystemOneRequest):
        return await infer(request)

    @app.post('/v1/rank', response_model=RankingResponse, dependencies=[Depends(authorize)])
    async def rank(request: RankRequest):
        return await infer(request, rank=True)

    return app


app = create_app()
