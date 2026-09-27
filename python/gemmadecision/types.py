"""Typed decision requests and responses; importing needs no ML runtime."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, AliasChoices, model_validator
from .constants import MODEL_ID, TEMPERATURE


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Choice(WireModel):
    type: Literal["choice"] = "choice"
    instructions: JsonValue
    criteria: dict[str, JsonValue] = Field(min_length=2, max_length=64)


class Noul(WireModel):
    type: Literal["noul"] = "noul"
    instructions: JsonValue
    criteria: dict[str, JsonValue] | None = None


class Score(WireModel):
    type: Literal["score"] = "score"
    instructions: JsonValue
    criteria: list[JsonValue] = Field(min_length=2, max_length=64)


Question = Annotated[Choice | Noul | Score, Field(discriminator="type")]


class SystemOneRequest(WireModel):
    state: JsonValue
    questions: dict[str, Question] = Field(min_length=1, max_length=32)
    model: str = MODEL_ID


class RankRequest(WireModel):
    state: JsonValue = Field(validation_alias=AliasChoices("state", "context"))
    candidates: dict[str, str] | list[str] = Field(validation_alias=AliasChoices("candidates", "answers"), min_length=2, max_length=64)
    question: JsonValue = ""
    model: str = MODEL_ID


class ChoiceAnswer(WireModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float
    scores: dict[str, float] = Field(default_factory=dict)


class NoulAnswer(WireModel):
    type: Literal["noul"] = "noul"
    noul: float
    probabilities: dict[str, float]
    confidence: float
    scores: dict[str, float] = Field(default_factory=dict)


class ScoreAnswer(WireModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[str, float]
    confidence: float
    scores: dict[str, float] = Field(default_factory=dict)


Answer = Annotated[ChoiceAnswer | NoulAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(WireModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_pairs: int = 0


class SystemOneResponse(WireModel):
    model: str = MODEL_ID
    answers: dict[str, Answer]
    usage: Usage
    temperature: float = TEMPERATURE
    probabilities_source: str = "softmax_ranking_scores_external_temperature"
    latency_ms: float = 0


class RankedCandidate(WireModel):
    rank: int
    candidate: str
    text: str
    score: float
    probability: float


class RankingResponse(WireModel):
    model: str = MODEL_ID
    ranked: list[RankedCandidate]
    usage: Usage
    temperature: float = TEMPERATURE
    probabilities_source: str = "softmax_ranking_scores_external_temperature"
    latency_ms: float = 0
