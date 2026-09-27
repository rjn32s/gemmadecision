"""Small sync/async clients. Connections are reused; decisions are never retried."""
import os
from typing import Any
import httpx
from pydantic import BaseModel
from .constants import DEFAULT_BASE_URL
from .types import Choice, ChoiceAnswer, SystemOneResponse, RankingResponse


def _json(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _options(timeout, extra_headers):
    options = {}
    if timeout is not None:
        options["timeout"] = timeout
    if extra_headers:
        options["headers"] = extra_headers
    return options


class DecisionClient:
    def __init__(self, base_url: str | None = None, *, api_key: str | None = None,
                 timeout: float = 60, transport=None):
        self.base_url = (base_url or os.getenv("GEMMADECISION_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        key = api_key if api_key is not None else os.getenv("GEMMADECISION_API_KEY")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport,
                                    headers={"Authorization": f"Bearer {key}"} if key else {},
                                    follow_redirects=False)

    def system_one(self, state: Any, questions: dict, *, timeout=None, extra_headers=None) -> SystemOneResponse:
        response = self._client.post("/v1/systemone", json=_json({"state": state, "questions": questions}),
                                     **_options(timeout, extra_headers))
        response.raise_for_status()
        return SystemOneResponse.model_validate(response.json())

    def decide(self, state: Any, candidates: dict[str, str], *, question: str = "Choose the best option.",
               timeout=None, extra_headers=None) -> ChoiceAnswer:
        answer = self.system_one(state, {"decision": Choice(instructions=question, criteria=candidates)},
                                 timeout=timeout, extra_headers=extra_headers).answers["decision"]
        assert isinstance(answer, ChoiceAnswer)
        return answer

    def rank(self, state: Any, candidates: dict[str, str] | list[str], *, question: Any = "",
             timeout=None, extra_headers=None) -> RankingResponse:
        response = self._client.post("/v1/rank", json=_json({"state": state, "candidates": candidates, "question": question}),
                                     **_options(timeout, extra_headers))
        response.raise_for_status()
        return RankingResponse.model_validate(response.json())

    def health(self) -> dict:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncDecisionClient:
    def __init__(self, base_url: str | None = None, *, api_key: str | None = None,
                 timeout: float = 60, transport=None):
        self.base_url = (base_url or os.getenv("GEMMADECISION_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        key = api_key if api_key is not None else os.getenv("GEMMADECISION_API_KEY")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, transport=transport,
                                        headers={"Authorization": f"Bearer {key}"} if key else {},
                                        follow_redirects=False)

    async def system_one(self, state: Any, questions: dict, *, timeout=None, extra_headers=None) -> SystemOneResponse:
        response = await self._client.post("/v1/systemone", json=_json({"state": state, "questions": questions}),
                                           **_options(timeout, extra_headers))
        response.raise_for_status()
        return SystemOneResponse.model_validate(response.json())

    async def decide(self, state: Any, candidates: dict[str, str], *, question: str = "Choose the best option.",
                     timeout=None, extra_headers=None) -> ChoiceAnswer:
        answer = (await self.system_one(state, {"decision": Choice(instructions=question, criteria=candidates)},
                                       timeout=timeout, extra_headers=extra_headers)).answers["decision"]
        assert isinstance(answer, ChoiceAnswer)
        return answer

    async def rank(self, state: Any, candidates: dict[str, str] | list[str], *, question: Any = "",
                   timeout=None, extra_headers=None) -> RankingResponse:
        response = await self._client.post("/v1/rank", json=_json({"state": state, "candidates": candidates, "question": question}),
                                           **_options(timeout, extra_headers))
        response.raise_for_status()
        return RankingResponse.model_validate(response.json())

    async def health(self) -> dict:
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    async def aclose(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
