# Compare a claim with evidence

Choose whether supplied evidence supports a claim, contradicts it, or leaves
it unresolved. Put the claim and evidence in the state and ask explicitly
about their relationship.

```python
from gemmadecision import Choice, ChoiceAnswer, DecisionEngine


RELATIONS = {
    "supported": "The supplied evidence establishes that the claim is true.",
    "contradicted": "The supplied evidence establishes that the claim is false.",
    "unknown": "The supplied evidence does not establish whether the claim is true or false.",
}


def classify_evidence(
    engine: DecisionEngine, evidence: str, claim: str
) -> ChoiceAnswer:
    response = engine.system_one(
        state={"evidence": evidence, "claim": claim},
        questions={
            "relation": Choice(
                instructions="Using only the supplied evidence, how does it relate to the claim?",
                criteria=RELATIONS,
            )
        },
    )
    answer = response.answers["relation"]
    assert isinstance(answer, ChoiceAnswer)
    return answer


def main() -> None:
    engine = DecisionEngine.from_pretrained()
    answer = classify_evidence(
        engine,
        evidence="The help page says invoices remain available for eighteen months.",
        claim="Invoices are deleted after three months.",
    )
    print(answer.choice)
    print(answer.probabilities)


if __name__ == "__main__":
    main()
```

`unknown` means insufficient evidence, not a low score for the other two
labels. Keep it as an explicit candidate. This recipe does not retrieve
documents or verify the truth of a source; it compares text that your
application supplies. For long documents, retrieve a relevant passage before
calling the model, respecting the input limit.
