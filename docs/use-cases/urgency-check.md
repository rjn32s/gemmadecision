# Ask a yes/no urgency question

Use `Noul` when the answer has two outcomes. Define the decision boundary in
plain language with `true` and `false` criteria.

```python
from gemmadecision import DecisionEngine, Noul, NoulAnswer


def assess_urgency(engine: DecisionEngine, message: str) -> NoulAnswer:
    response = engine.system_one(
        state=message,
        questions={
            "urgent": Noul(
                instructions="Does this support request need attention today?",
                criteria={
                    "true": "The message gives a deadline today or says work is currently blocked.",
                    "false": "The request is routine, has a later deadline, or gives no evidence of urgency.",
                },
            )
        },
    )
    answer = response.answers["urgent"]
    assert isinstance(answer, NoulAnswer)
    return answer


def main() -> None:
    engine = DecisionEngine.from_pretrained()
    answer = assess_urgency(
        engine,
        "The export stops halfway. I need the file for a meeting this afternoon.",
    )
    print({"yes_probability": answer.noul, "verdict_at_0_5": answer.noul >= 0.5})


if __name__ == "__main__":
    main()
```

`.noul` is the probability assigned to `true`, derived from the two ranking
scores. The example uses a 0.5 cutoff to choose the higher-scoring outcome.
That cutoff does not establish an acceptable missed-urgency rate; choose an
application threshold using held-out labeled messages.

For more than two urgency levels, use an [ordered rubric](urgency-rubric.md).
