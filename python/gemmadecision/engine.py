"""One typed decision contract across batched PyTorch and vLLM inference."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

from . import _clm_schema as schema
from .constants import MODEL_ID, MODEL_REPO, TEMPERATURE
from .types import (SystemOneRequest, SystemOneResponse, RankRequest, RankingResponse,
                    Choice, ChoiceAnswer, NoulAnswer, ScoreAnswer, Usage, RankedCandidate)


@dataclass
class PreparedQuestion:
    name: str
    kind: str
    labels: list[str]
    candidates: list[str]
    texts: list[str]
    tokens: list[int]


@dataclass
class PreparedRequest:
    questions: list[PreparedQuestion]
    texts: list[str]
    tokens: list[int]


def render_state(state: Any, instructions: Any) -> str:
    # Preserve the v4 serving recipe, including JSON-object strings.
    if isinstance(state, str):
        try:
            parsed = json.loads(state)
            if isinstance(parsed, (dict, list)):
                state = parsed
        except ValueError:
            pass
    return schema.state_text(state, instructions)


def probabilities(scores: list[float]) -> list[float]:
    if not scores or any(not math.isfinite(score) for score in scores):
        raise RuntimeError("The inference backend returned non-finite or empty scores")
    scaled = [float(x) / TEMPERATURE for x in scores]
    maximum = max(scaled)
    values = [math.exp(x - maximum) for x in scaled]
    total = math.fsum(values)
    return [value / total for value in values]


class DecisionEngine:
    """Local in-process engine. Model loading is explicit; no generative API."""
    def __init__(self, backend, *, cache_size: int = 0):
        if not 0 <= cache_size <= 1_000_000:
            raise ValueError("cache_size must be between 0 and 1000000")
        self.backend = backend
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(cls, model_path: str | Path | None = None, *, device="auto", backend="torch",
                        strict=False, max_batch_tokens=8192, max_batch_size=32, cache_size=0,
                        offline=False, gpu_memory_utilization=.2):
        if model_path is None:
            from .model import download_model
            model_path = download_model(offline=offline)
        if backend == "torch":
            from .backends.torch import TorchBackend
            instance = TorchBackend(model_path, device=device, strict=strict,
                                    max_batch_tokens=max_batch_tokens, max_batch_size=max_batch_size)
        elif backend == "vllm":
            if device not in {"auto", "cuda"}:
                raise ValueError("The vLLM backend requires a supported CUDA GPU")
            if strict:
                raise ValueError("Strict singleton reference mode is available only with --backend torch")
            from .backends.vllm import VLLMBackend
            instance = VLLMBackend(model_path, gpu_memory_utilization=gpu_memory_utilization,
                                   max_num_seqs=max_batch_size)
        else:
            raise ValueError("backend must be torch or vllm")
        return cls(instance, cache_size=cache_size)

    def prepare(self, request: SystemOneRequest) -> PreparedRequest:
        # Hugging Face fast-tokenizer padding configuration is mutable. Keep
        # preflight tokenization serialized with batch tokenization/inference.
        with self._lock:
            return self._prepare(request)

    def _prepare(self, request: SystemOneRequest) -> PreparedRequest:
        if request.model not in {MODEL_ID, MODEL_REPO, "GemmaDecision-270M", "gemmadecision"}:
            raise ValueError("Unknown model; this server serves the pinned GemmaDecision-270M v0.4.0")
        result = []
        flat_texts, flat_tokens = [], []
        for name, question in request.questions.items():
            wire = question.model_dump(mode="json")
            if wire["type"] == "noul":
                criteria = wire.get("criteria") or {}
                if set(criteria) - {"false", "true", "no", "yes"}:
                    raise ValueError("Noul criteria must use false/true or no/yes")
                normalized = {}
                for native, alias in (("false", "no"), ("true", "yes")):
                    if native in criteria and alias in criteria:
                        raise ValueError("Noul criteria have ambiguous aliases")
                    if native in criteria:
                        normalized[native] = criteria[native]
                    elif alias in criteria:
                        normalized[native] = criteria[alias]
                wire["criteria"] = normalized
            labels, candidates = schema.candidates(wire)
            if not 2 <= len(candidates) <= 64 or len(set(candidates)) != len(candidates):
                raise ValueError("Provide 2–64 distinct candidate descriptions")
            if any(not candidate.strip() for candidate in candidates):
                raise ValueError("Candidate descriptions must be nonempty")
            state_text = render_state(request.state, wire["instructions"])
            if not state_text.strip():
                raise ValueError("Provide a nonempty state or question")
            count = self.backend.count_tokens(state_text)
            if count > self.backend.max_state_tokens:
                raise ValueError(f"State/question has {count} tokens; the limit is {self.backend.max_state_tokens}")
            texts, counts = [], []
            for candidate in candidates:
                n = self.backend.count_tokens(candidate)
                if n > self.backend.max_action_tokens:
                    raise ValueError(f"Candidate has {n} tokens; the limit is {self.backend.max_action_tokens}")
                text = state_text + "\n\nCandidate action:\n" + candidate
                token_count = self.backend.count_tokens(text)
                if token_count > self.backend.context_limit:
                    raise ValueError("Joint input exceeds the encoder context limit")
                if token_count > getattr(self.backend, "max_batch_tokens", self.backend.context_limit):
                    raise ValueError("Joint input exceeds the configured batch token limit")
                texts.append(text)
                counts.append(token_count)
            result.append(PreparedQuestion(name, wire["type"], labels, candidates, texts, counts))
            flat_texts.extend(texts)
            flat_tokens.extend(counts)
        if len(flat_texts) > 256:
            raise ValueError("A request may contain at most 256 candidate pairs across all questions")
        return PreparedRequest(result, flat_texts, flat_tokens)

    def score_many(self, plans: list[PreparedRequest]) -> list[tuple[list[float], Usage]]:
        """Deduplicate identical full joint inputs; never cache candidates independently."""
        with self._lock:
            texts = [text for plan in plans for text in plan.texts]
            keys = [hashlib.sha256(text.encode()).digest() for text in texts]
            known, missing = {}, OrderedDict()
            for key, text in zip(keys, texts):
                if key in self._cache:
                    known[key] = self._cache[key]
                    self._cache.move_to_end(key)
                elif key not in missing:
                    missing[key] = text
            scores = self.backend.score_pairs(list(missing.values())) if missing else []
            if len(scores) != len(missing) or any(not math.isfinite(float(score)) for score in scores):
                raise RuntimeError("Backend returned an invalid score vector")
            computed = dict(zip(missing, map(float, scores)))
            if self.cache_size:
                self._cache.update(computed)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            results = []
            offset = 0
            # Count each actually encoded pair once in this batch. Repeated
            # state/question tokens are counted in every distinct joint pair.
            spent = set(known)
            for plan in plans:
                plan_keys = keys[offset:offset + len(plan.texts)]
                values, tokens, hits = [], 0, 0
                for key, token_count in zip(plan_keys, plan.tokens):
                    values.append(known[key] if key in known else computed[key])
                    if key in spent:
                        hits += 1
                    else:
                        tokens += token_count
                        spent.add(key)
                results.append((values, Usage(input_tokens=tokens, total_tokens=tokens, cached_pairs=hits)))
                offset += len(plan.texts)
            return results

    def finish(self, plan, scored, *, latency_ms=0) -> SystemOneResponse:
        values, usage = scored
        offset, answers = 0, {}
        for question in plan.questions:
            n = len(question.labels)
            scores = values[offset:offset + n]
            probs = probabilities(scores)
            distribution = dict(zip(question.labels, probs))
            common = dict(probabilities=distribution, confidence=max(probs), scores=dict(zip(question.labels, scores)))
            if question.kind == "choice":
                answer = ChoiceAnswer(choice=question.labels[max(range(n), key=probs.__getitem__)], **common)
            elif question.kind == "noul":
                answer = NoulAnswer(noul=distribution["true"], **common)
            else:
                answer = ScoreAnswer(score=math.fsum(i * value for i, value in enumerate(probs)), **common)
            answers[question.name] = answer
            offset += n
        return SystemOneResponse(answers=answers, usage=usage, latency_ms=latency_ms)

    def system_one(self, state: Any, questions: dict) -> SystemOneResponse:
        started = time.perf_counter()
        plan = self.prepare(SystemOneRequest(state=state, questions=questions))
        return self.finish(plan, self.score_many([plan])[0], latency_ms=(time.perf_counter() - started) * 1000)

    def decide(self, state: Any, candidates: dict[str, str], *, question="Choose the best option.") -> ChoiceAnswer:
        return self.system_one(state, {"decision": Choice(instructions=question, criteria=candidates)}).answers["decision"]

    def prepare_rank(self, request: RankRequest):
        candidates = request.candidates
        if isinstance(candidates, list):
            candidates = {str(i): text for i, text in enumerate(candidates)}
        return self.prepare(SystemOneRequest(state=request.state, model=request.model,
                            questions={"rank": Choice(instructions=request.question, criteria=candidates)}))

    def finish_rank(self, plan, scored, *, latency_ms=0):
        scores, usage = scored
        question = plan.questions[0]
        probs = probabilities(scores)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return RankingResponse(ranked=[RankedCandidate(rank=rank + 1, candidate=question.labels[i],
                                text=question.candidates[i], score=scores[i], probability=probs[i])
                                for rank, i in enumerate(order)], usage=usage, latency_ms=latency_ms)

    def rank(self, state, candidates, *, question=""):
        started = time.perf_counter()
        plan = self.prepare_rank(RankRequest(state=state, candidates=candidates, question=question))
        return self.finish_rank(plan, self.score_many([plan])[0], latency_ms=(time.perf_counter() - started) * 1000)

    def clear_cache(self):
        with self._lock:
            self._cache.clear()

    def metadata(self):
        extra = getattr(self.backend, "metadata", {})
        if callable(extra):
            extra = extra()
        return {"model": MODEL_ID, "backend": type(self.backend).__name__, "cache_size": self.cache_size,
                "temperature": TEMPERATURE, "max_state_tokens": self.backend.max_state_tokens,
                "max_action_tokens": self.backend.max_action_tokens, **(extra if isinstance(extra, dict) else {})}
