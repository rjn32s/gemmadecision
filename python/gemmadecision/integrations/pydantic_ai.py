"""Native PydanticAI decision-model support via ``gemmadecision[pydantic-ai]``.

GemmaDecision ranks supplied alternatives. PydanticAI's DecisionModel translates
finite typed outputs into those alternatives; it does not ask this model to
generate JSON or prose.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, ClassVar, TYPE_CHECKING

import httpx
from pydantic import ValidationError

try:
    from pydantic_ai.exceptions import (
        ModelAPIError,
        ModelHTTPError,
        UnexpectedModelBehavior,
        UserError,
    )
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.decision import (
        ChoiceAnswer,
        ChoiceQuestion,
        DecisionModel,
        DecisionModelSettings,
        DecisionRequest,
        DecisionResponse,
        NoulAnswer,
        NoulQuestion,
        ScoreAnswer,
        ScoreQuestion,
    )
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.usage import RequestUsage
except ImportError as exc:
    raise ImportError(
        "Install the optional integration: pip install 'gemmadecision[pydantic-ai]' "
        "(pydantic-ai-slim >= 2.51, < 3)."
    ) from exc

from gemmadecision.client import AsyncDecisionClient
from gemmadecision.constants import MODEL_ID

if TYPE_CHECKING:
    from gemmadecision.engine import DecisionEngine

__all__ = ["GemmaDecisionModel"]


class GemmaDecisionModel(DecisionModel["AsyncDecisionClient | _LocalDecisionClient"]):
    """Use GemmaDecision locally or through a server as a PydanticAI model.

    Use ``bool``, ``Literal``, ``Enum`` or a Pydantic model of supported decision
    fields as the agent's ``output_type``. Free-form text is unsupported. The
    reported probabilities are softmax-transformed ranking scores, not a
    guarantee of correctness on the application's data.

    Pass an existing ``AsyncDecisionClient`` to reuse its connection pool. The
    caller owns an injected client; the model closes only a client it creates.
    ``GemmaDecisionModel.local()`` needs no server and lazily shares the default
    process-local engine with ``gemmadecision.decide`` and ``gemmadecision.rank``.
    """

    max_choice_options: ClassVar[int] = 64
    max_score_levels: ClassVar[int] = 64

    def __init__(
        self,
        model_name: str = "rajan2k/GemmaDecision-270M",
        *,
        base_url: str = "http://127.0.0.1:8700",
        api_key: str | None = None,
        client: AsyncDecisionClient | _LocalDecisionClient | None = None,
        settings: ModelSettings | None = None,
    ) -> None:
        self._model_name = model_name
        self._owns_client = client is None
        self._client = client or AsyncDecisionClient(base_url=base_url, api_key=api_key)
        super().__init__(settings=settings, profile={"context_window": 2048})

    @classmethod
    def local(
        cls,
        *,
        engine: DecisionEngine | None = None,
        settings: ModelSettings | None = None,
    ) -> GemmaDecisionModel:
        """Create a local typed model without loading weights or starting HTTP.

        First use loads the default shared engine in a worker thread. Inject an
        existing ``DecisionEngine`` for explicit device/backend configuration.
        The process or caller owns the engine; closing this model does not unload
        a shared model from other users in the same process.
        """
        return cls(model_name=MODEL_ID, client=_LocalDecisionClient(engine), settings=settings)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return "gemmadecision"

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    @property
    def client(self) -> AsyncDecisionClient | _LocalDecisionClient:
        return self._client

    async def __aenter__(self) -> GemmaDecisionModel:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close this model's own client, leaving injected clients open."""
        if self._owns_client:
            await self._client.aclose()

    async def decide(
        self, request: DecisionRequest, model_settings: DecisionModelSettings
    ) -> DecisionResponse:
        """Forward typed questions through the local engine or one HTTP request."""
        if model_settings.get("extra_body") is not None:
            raise UserError("GemmaDecision does not accept model_settings['extra_body'].")
        if not 1 <= len(request.questions) <= 32:
            raise UserError("GemmaDecision accepts 1 to 32 questions in one request.")

        questions: dict[str, dict[str, Any]] = {}
        for name, question in request.questions.items():
            if not isinstance(question, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
                raise UserError(f"Unsupported decision question {name!r}.")
            wire = {key: value for key, value in asdict(question).items() if value is not None}
            wire["instructions"] = question.instructions
            if isinstance(question, NoulQuestion) and "criteria" in wire:
                wire["criteria"] = {
                    key: value for key, value in wire["criteria"].items() if value is not None
                }
            questions[name] = wire

        options: dict[str, Any] = {}
        if "timeout" in model_settings:
            timeout = model_settings["timeout"]
            # PydanticAI 2 uses HTTPX2; accept its structurally equivalent timeout
            # without passing that unrelated class into the HTTPX client.
            if not isinstance(timeout, (int, float, httpx.Timeout)):
                try:
                    timeout = httpx.Timeout(
                        connect=timeout.connect,
                        read=timeout.read,
                        write=timeout.write,
                        pool=timeout.pool,
                    )
                except AttributeError as exc:
                    raise UserError("Unsupported GemmaDecision timeout value.") from exc
            options["timeout"] = timeout
        if "extra_headers" in model_settings:
            options["extra_headers"] = model_settings["extra_headers"]
        try:
            response = await self.client.system_one(
                state=request.state, questions=questions, **options
            )
        except httpx.HTTPStatusError as exc:
            raise ModelHTTPError(
                status_code=exc.response.status_code,
                model_name=self.model_name,
                body=exc.response.text,
                headers=exc.response.headers,
            ) from exc
        except httpx.RequestError as exc:
            raise ModelAPIError(self.model_name, str(exc)) from exc
        except (ValidationError, json.JSONDecodeError) as exc:
            raise UnexpectedModelBehavior("Invalid GemmaDecision protocol response.", str(exc)) from exc

        try:
            raw_answers = _field(response, "answers")
            if set(raw_answers) != set(request.questions):
                raise ValueError("The server did not return exactly the requested questions.")
            answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = {}
            for name, question in request.questions.items():
                answer = raw_answers[name]
                if _field(answer, "type") != question.type:
                    raise ValueError(f"Answer type does not match question {name!r}.")
                if isinstance(question, NoulQuestion):
                    probs = _probabilities(answer, {"false", "true"})
                    noul = float(_field(answer, "noul"))
                    if not math.isclose(noul, probs["true"], rel_tol=1e-6, abs_tol=1e-6):
                        raise ValueError("The yes probability disagrees with the distribution.")
                    answers[name] = NoulAnswer(noul=noul)
                elif isinstance(question, ChoiceQuestion):
                    probs = _probabilities(answer, set(question.criteria))
                    choice = _field(answer, "choice")
                    if choice not in probs or probs[choice] != max(probs.values()):
                        raise ValueError("The selected choice must have the highest probability.")
                    answers[name] = ChoiceAnswer(
                        choice=choice, confidence=probs[choice], probabilities=probs
                    )
                else:
                    probs = _probabilities(answer, {str(i) for i in range(len(question.criteria))})
                    score = float(_field(answer, "score"))
                    expected = sum(int(level) * prob for level, prob in probs.items())
                    if not math.isclose(score, expected, rel_tol=1e-6, abs_tol=1e-6):
                        raise ValueError("The rubric score must equal its distribution's expectation.")
                    answers[name] = ScoreAnswer(
                        score=score,
                        confidence=max(probs.values()),
                        probabilities={int(level): prob for level, prob in probs.items()},
                        legend=dict(enumerate(question.criteria)),
                    )
            usage = _field(response, "usage", {})
            return DecisionResponse(
                answers=answers,
                model_name=_field(response, "model", self.model_name),
                usage=RequestUsage(
                    input_tokens=_field(usage, "input_tokens", 0),
                    output_tokens=_field(usage, "output_tokens", 0),
                ),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise UnexpectedModelBehavior(f"Invalid GemmaDecision answer: {exc}") from exc

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await super().request(messages, model_settings, model_request_parameters)
        response.provider_details = {
            **(response.provider_details or {}),
            "probabilities_source": "softmax_scores",
            "confidence_note": "Ranking probabilities; validate calibration on your own data.",
        }
        return response


class _LocalDecisionClient:
    """Async facade over a shared synchronous engine; never starts a server."""

    base_url = "local://gemmadecision"

    def __init__(self, engine: DecisionEngine | None = None) -> None:
        self._engine = engine

    async def aclose(self) -> None:
        """The process or caller owns the shared engine; no HTTP pool to close."""

    async def system_one(self, state, questions, *, timeout=None, extra_headers=None):
        if extra_headers:
            raise UserError("extra_headers applies to remote GemmaDecision servers, not local inference.")
        if isinstance(timeout, httpx.Timeout):
            timeout = timeout.read
        if timeout is not None and (
            not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0
        ):
            raise UserError("A local inference timeout must be a positive finite number of seconds.")

        stopped = threading.Event()
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._run, state, questions, stopped), timeout=timeout
            )
        except asyncio.TimeoutError as exc:
            stopped.set()
            raise ModelAPIError(
                MODEL_ID, "Local inference timed out; an already running model operation may finish in the background."
            ) from exc
        except asyncio.CancelledError:
            stopped.set()
            raise

    def _run(self, state, questions, stopped):
        try:
            engine = self._engine
            if engine is None:
                from gemmadecision.simple import get_engine
                engine = get_engine()
        except ImportError as exc:
            raise UserError(
                "Local inference requires the model runtime: pip install gemmadecision."
            ) from exc
        except Exception as exc:
            raise ModelAPIError(MODEL_ID, f"Could not load the local model: {exc}") from exc
        if stopped.is_set():
            raise RuntimeError("Local request was cancelled before inference.")
        from gemmadecision.types import SystemOneRequest
        try:
            plan = engine.prepare(SystemOneRequest(state=state, questions=questions))
        except ValueError as exc:
            raise UserError(f"Invalid local GemmaDecision request: {exc}") from exc
        except Exception as exc:
            raise ModelAPIError(MODEL_ID, f"Local request preparation failed: {exc}") from exc
        if stopped.is_set():
            raise RuntimeError("Local request was cancelled before inference.")
        try:
            return engine.finish(plan, engine.score_many([plan])[0])
        except Exception as exc:
            raise ModelAPIError(MODEL_ID, f"Local inference failed: {exc}") from exc


def _field(value: Any, name: str, default: Any = ...) -> Any:
    if isinstance(value, Mapping):
        return value[name] if default is ... else value.get(name, default)
    return getattr(value, name) if default is ... else getattr(value, name, default)


def _probabilities(answer: Any, labels: set[str]) -> dict[str, float]:
    raw = _field(answer, "probabilities")
    if not isinstance(raw, Mapping):
        raise ValueError("Probabilities must be a mapping.")
    probabilities = {str(label): float(prob) for label, prob in raw.items()}
    if len(probabilities) != len(raw) or set(probabilities) != labels:
        raise ValueError("Probability labels do not match the question's options.")
    if not all(math.isfinite(prob) and 0 <= prob <= 1 for prob in probabilities.values()):
        raise ValueError("Probabilities must be finite values between zero and one.")
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Probabilities do not sum to one.")
    return probabilities
