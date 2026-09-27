"""Runnable decision recipes using the published gemmadecision 0.1 API.

    python examples/recipes.py --list
    python examples/recipes.py --recipe typed-agent --device cpu

Listing or importing this file never loads a model. Running a recipe explicitly
loads one shared local engine. First use can download the published weights.
These examples show API mechanics; their outputs are not accuracy guarantees.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from typing import Callable, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field

from gemmadecision import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer

if TYPE_CHECKING:
    from gemmadecision.engine import DecisionEngine


TICKET = "I was charged twice for one order. Please help me resolve this today."
TEAMS = {
    "billing": "Help with duplicate charges, invoices, payments, and refunds.",
    "account": "Help with signing in, passwords, and account details.",
    "technical": "Help with errors, broken features, and product operation.",
    "review": "Ask a person to review requests outside these categories or lacking enough information.",
}


class Triage(BaseModel):
    team: Literal["billing", "account", "technical", "review"] = Field(
        description="Which support team should handle this request? Use review when the request does not fit."
    )
    urgent: bool = Field(description="Does the customer explicitly ask for attention today?")


def direct_choice(engine: DecisionEngine, state: str = TICKET) -> dict:
    """Choose a stable label from descriptions supplied by the application."""
    answer = engine.decide(state, TEAMS, question="Choose the support team for this customer request.")
    return {"team": answer.choice, "probabilities": answer.probabilities}


def ranking(engine: DecisionEngine, state: str = TICKET) -> dict:
    """Inspect alternatives and raw scores, retaining the application's labels."""
    result = engine.rank(state, TEAMS, question="Rank the teams by suitability for this request.")
    return {
        "ranked": [item.model_dump(mode="json") for item in result.ranked],
        "probabilities_source": result.probabilities_source,
    }


def multiple_questions(engine: DecisionEngine, state: str = TICKET) -> dict:
    """Ask a choice, a yes/no question, and an ordered rubric in one call."""
    result = engine.system_one(state, {
        "team": Choice(instructions="Choose the support team.", criteria=TEAMS),
        "urgent": Noul(instructions="The customer explicitly asks for attention today."),
        "detail": Score(
            instructions="How much useful detail has the customer supplied?",
            criteria=[
                "The request contains too little detail to identify the problem.",
                "The request identifies the problem but lacks supporting details.",
                "The request identifies the problem and includes supporting details.",
            ],
        ),
    })
    team, urgent, detail = (result.answers[name] for name in ("team", "urgent", "detail"))
    assert isinstance(team, ChoiceAnswer)
    assert isinstance(urgent, NoulAnswer)
    assert isinstance(detail, ScoreAnswer)
    return {
        "team": team.choice,
        "urgent_probability": urgent.noul,
        "detail_score_0_to_2": detail.score,
        "answers": {name: answer.model_dump(mode="json") for name, answer in result.answers.items()},
    }


def typed_agent(engine: DecisionEngine, state: str = TICKET) -> dict:
    """Get a Pydantic model locally; no HTTP server or generative LLM is needed."""
    from pydantic_ai import Agent
    from gemmadecision import GemmaDecisionModel

    agent = Agent(GemmaDecisionModel.local(engine=engine), output_type=Triage)
    result = agent.run_sync(state)
    return result.output.model_dump(mode="json")


TOOL_OPTIONS = {
    "search_docs": "Search the public product documentation for an explanation.",
    "lookup_order": "Read the existing order record to check its payment status.",
    "ask_customer": "Ask the customer for missing information needed to continue.",
    "human_review": "Send the recommendation to a human reviewer without executing a tool.",
}


def tool_recommendation(
    engine: DecisionEngine,
    state: str = TICKET,
    *,
    allowed_tools: frozenset[str] = frozenset({"search_docs", "lookup_order", "ask_customer"}),
) -> dict:
    """Recommend a bounded label, then enforce application permissions.

    This example never invokes tools. The host application owns actual execution
    and its permission checks; a model-selected label cannot grant permission.
    """
    answer = engine.decide(state, TOOL_OPTIONS, question="Which next support action would help most?")
    permitted = allowed_tools.intersection(TOOL_OPTIONS.keys() - {"human_review"})
    recommended = answer.choice if answer.choice in permitted else "human_review"
    return {
        "model_choice": answer.choice,
        "recommendation": recommended,
        "allowed_by_application": answer.choice in permitted,
        "executed": False,
    }


def evidence_relation(engine: DecisionEngine) -> dict:
    """Choose whether supplied evidence supports, contradicts, or leaves a claim open."""
    state = {
        "evidence": "The shop opens at 09:00 every weekday. Today is Tuesday.",
        "claim": "The shop opens at 09:00 today.",
    }
    answer = engine.decide(state, {
        "supports": "The supplied evidence supports the claim.",
        "contradicts": "The supplied evidence contradicts the claim.",
        "unknown": "The supplied evidence is insufficient to determine whether the claim is true.",
    }, question="Classify the claim using only the supplied evidence.")
    return {"relation": answer.choice, "probabilities": answer.probabilities}


def explicit_fallback(
    engine: DecisionEngine,
    state: str = "Can you help me with something?",
    *,
    minimum_probability: float | None = None,
) -> dict:
    """Offer review and optionally apply an application-validated cutoff.

    Scores produce a ranking distribution, not a correctness guarantee. Choose
    a cutoff on held-out application data; this example supplies no default.
    """
    if minimum_probability is not None and not 0 <= minimum_probability <= 1:
        raise ValueError("minimum_probability must be between zero and one")
    answer = engine.decide(state, {
        "billing": TEAMS["billing"],
        "technical": TEAMS["technical"],
        "human_review": "Ask a person to clarify an ambiguous request or handle a request outside billing and technical support.",
    }, question="Choose a destination. Use human_review when the request does not identify a supported issue.")
    below_cutoff = minimum_probability is not None and answer.probabilities[answer.choice] < minimum_probability
    destination = "human_review" if below_cutoff else answer.choice
    return {
        "model_choice": answer.choice,
        "destination": destination,
        "needs_review": destination == "human_review",
        "below_application_cutoff": below_cutoff,
        "probabilities": answer.probabilities,
    }


@dataclass(frozen=True)
class Recipe:
    description: str
    run: Callable[[DecisionEngine], dict]


RECIPES = {
    "direct-choice": Recipe("Choose one application label", direct_choice),
    "ranking": Recipe("Rank candidates and inspect scores", ranking),
    "multiple-questions": Recipe("Choice, Noul, and Score in one call", multiple_questions),
    "typed-agent": Recipe("Native local PydanticAI with a finite typed output", typed_agent),
    "tool-recommendation": Recipe("Recommend a permitted tool label without executing it", tool_recommendation),
    "evidence-relation": Recipe("Classify a claim against supplied evidence", evidence_relation),
    "explicit-fallback": Recipe("Include a human-review candidate", explicit_fallback),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true", help="List recipes without loading a model")
    selection.add_argument("--recipe", choices=RECIPES, help="Run one recipe using a shared local model")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args(argv)
    if args.list:
        for name, recipe in RECIPES.items():
            print(f"{name}: {recipe.description}")
        return 0

    from gemmadecision import DecisionEngine

    engine = DecisionEngine.from_pretrained(device=args.device)
    result = RECIPES[args.recipe].run(engine)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
