# Fall back to human review

Give the model a review option, then apply your own minimum-probability policy.
An explicit review option handles unclear requests; a threshold can also
catch a close or weak preference between the offered routes.

```python
from dataclasses import asdict, dataclass

from gemmadecision import Choice, ChoiceAnswer, DecisionEngine


ROUTES = {
    "billing": "A clear request about invoices, charges or refunds.",
    "technical": "A clear request about a software feature that is not working.",
    "review": "An unclear, mixed or unrelated request that a person should inspect.",
}


@dataclass(frozen=True)
class RoutingDecision:
    suggested_route: str
    accepted_route: str | None
    needs_review: bool
    top_probability: float


def route_with_review(
    engine: DecisionEngine,
    ticket: str,
    *,
    minimum_probability: float,
) -> RoutingDecision:
    if not 0 <= minimum_probability <= 1:
        raise ValueError("minimum_probability must be between zero and one")
    response = engine.system_one(
        state=ticket,
        questions={
            "route": Choice(
                instructions="Which route is best supported by the request?",
                criteria=ROUTES,
            )
        },
    )
    answer = response.answers["route"]
    assert isinstance(answer, ChoiceAnswer)
    top_probability = answer.probabilities[answer.choice]
    review = answer.choice == "review" or top_probability < minimum_probability
    return RoutingDecision(
        suggested_route=answer.choice,
        accepted_route=None if review else answer.choice,
        needs_review=review,
        top_probability=top_probability,
    )


def main() -> None:
    engine = DecisionEngine.from_pretrained()
    # Demonstration value only: select a threshold using your own held-out data.
    result = route_with_review(
        engine,
        "Something changed after the last payment and now I cannot finish the setup.",
        minimum_probability=0.70,
    )
    print(asdict(result))


if __name__ == "__main__":
    main()
```

A top probability of 0.70 is not a claim of 70% correctness. The example
cutoff is an application setting, not a recommended production threshold.
Choose it using held-out tickets: measure the error rate among accepted
routes and the share sent to review. Keep the candidate set fixed while
selecting that threshold, since probabilities depend on the alternatives.

The function returns a review flag; it does not send a message or create a
ticket. If `needs_review` is true, your application can add the item to a
review queue. Preserve the suggested route and scores for analysis without
treating them as the reviewer's decision.
