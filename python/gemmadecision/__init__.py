"""Typed decisions without importing torch, vLLM, or loading a model."""
from .client import AsyncDecisionClient, DecisionClient
from .types import Choice, Noul, Score, ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse, RankingResponse
from .constants import MODEL_ID, MODEL_REVISION

__version__ = "0.1.0"
__all__ = ["decide", "rank", "DecisionClient", "AsyncDecisionClient", "DecisionEngine", "GemmaDecisionModel", "Choice", "Noul", "Score",
           "ChoiceAnswer", "NoulAnswer", "ScoreAnswer", "SystemOneResponse", "RankingResponse"]


def __getattr__(name):
    if name == "DecisionEngine":
        from .engine import DecisionEngine
        return DecisionEngine
    if name == "GemmaDecisionModel":
        from .integrations.pydantic_ai import GemmaDecisionModel
        return GemmaDecisionModel
    raise AttributeError(name)


def decide(state, choices, *, question="Choose the best option.") -> str:
    """Choose locally in one call; first use downloads/loads the pinned model."""
    from .simple import get_engine
    from .types import RankRequest
    if isinstance(choices, (list, tuple)):
        if any(not isinstance(text, str) for text in choices):
            raise ValueError("Choices must be strings or a mapping of labels to descriptions")
        if len(set(choices)) != len(choices):
            raise ValueError("Choices must be distinct")
        choices = {text: text for text in choices}
    choices = RankRequest(state=state, candidates=choices, question=question).candidates
    if len(set(choices.values())) != len(choices) or any(not text.strip() for text in choices.values()):
        raise ValueError("Choice descriptions must be distinct and nonempty")
    return get_engine().decide(state, choices, question=question).choice


def rank(state, choices, *, question=""):
    """Detailed local ranking with lazy model loading, no server required."""
    from .simple import get_engine
    from .types import RankRequest
    choices = RankRequest(state=state, candidates=choices, question=question).candidates
    return get_engine().rank(state, choices, question=question)
