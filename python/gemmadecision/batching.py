"""Bounded cross-request microbatching; one model owner per server process."""
import asyncio
from dataclasses import dataclass
import time


class QueueFullError(RuntimeError):
    pass


@dataclass
class Pending:
    plan: object
    future: asyncio.Future
    prepared: bool = True
    rank: bool = False


class MicroBatcher:
    def __init__(self, engine, *, queue_size=128, batch_requests=32, wait_ms=1.0, timeout_s=60):
        if queue_size < 1 or batch_requests < 1 or not 0 <= wait_ms <= 100 or timeout_s <= 0:
            raise ValueError("Invalid queue or microbatch limits")
        self.engine = engine
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.batch_requests = batch_requests
        self.wait_ms = wait_ms
        self.timeout_s = timeout_s
        self.task = None
        self.closed = False
        self.completed_requests = 0
        self.model_batches = 0

    async def start(self):
        if self.task is not None:
            raise RuntimeError("Batcher already started")
        self.task = asyncio.create_task(self._run(), name="gemmadecision-inference")

    async def submit(self, plan):
        return await self._enqueue(plan, prepared=True)

    async def submit_request(self, request, *, rank=False):
        """Bound preparation, waiting and inference inside one queue/timeout."""
        return await self._enqueue(request, prepared=False, rank=rank)

    async def _enqueue(self, plan, *, prepared, rank=False):
        if self.closed or self.task is None or self.task.done():
            raise RuntimeError("Inference worker is unavailable")
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait(Pending(plan, future, prepared, rank))
        except asyncio.QueueFull as exc:
            raise QueueFullError("Inference queue is full; try again later") from exc
        return await asyncio.wait_for(future, timeout=self.timeout_s)

    def _process(self, pending):
        outputs, plans, positions = [None] * len(pending), [], []
        for index, item in enumerate(pending):
            if item.future.cancelled():
                outputs[index] = asyncio.CancelledError()
                continue
            try:
                plan = item.plan if item.prepared else (
                    self.engine.prepare_rank(item.plan) if item.rank else self.engine.prepare(item.plan))
            except Exception as exc:
                outputs[index] = exc
            else:
                plans.append(plan)
                positions.append(index)
        # Preparation can outlive a caller's deadline. Drop those requests
        # before scheduling any GPU work (in-flight GPU calls cannot be undone).
        kept = [(plan, index) for plan, index in zip(plans, positions)
                if not pending[index].future.done()]
        plans = [plan for plan, _ in kept]
        positions = [index for _, index in kept]
        if plans:
            try:
                scored = self.engine.score_many(plans)
                if len(scored) != len(plans):
                    raise RuntimeError("Inference worker returned an incomplete batch")
                for index, plan, score in zip(positions, plans, scored):
                    outputs[index] = score if pending[index].prepared else (plan, score)
            except Exception as exc:
                for index in positions:
                    outputs[index] = exc
        return outputs

    async def _run(self):
        active = []
        try:
            while True:
                active = [await self.queue.get()]
                deadline = time.monotonic() + self.wait_ms / 1000
                while len(active) < self.batch_requests:
                    try:
                        active.append(self.queue.get_nowait())
                        continue
                    except asyncio.QueueEmpty:
                        pass
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        active.append(await asyncio.wait_for(self.queue.get(), remaining))
                    except asyncio.TimeoutError:
                        break
                live = [item for item in active if not item.future.done()]
                if live:
                    try:
                        outputs = await asyncio.to_thread(self._process, live)
                        if len(outputs) != len(live):
                            raise RuntimeError("Inference worker returned an incomplete batch")
                        self.model_batches += 1
                        for item, value in zip(live, outputs):
                            if not item.future.done():
                                if isinstance(value, BaseException):
                                    item.future.set_exception(value)
                                else:
                                    item.future.set_result(value)
                                    self.completed_requests += 1
                    except Exception as exc:
                        for item in live:
                            if not item.future.done():
                                item.future.set_exception(exc)
                for item in active:
                    self.queue.task_done()
                active = []
        finally:
            for item in active:
                if not item.future.done():
                    item.future.set_exception(RuntimeError("Server is shutting down"))
                self.queue.task_done()

    async def close(self):
        self.closed = True
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if not item.future.done():
                item.future.set_exception(RuntimeError("Server is shutting down"))
            self.queue.task_done()
