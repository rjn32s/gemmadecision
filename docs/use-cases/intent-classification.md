# Classify an intent

Map a message to a small set of operations your application understands.
`Choice` returns the selected key plus a distribution over the alternatives.

```python
from gemmadecision import Choice, ChoiceAnswer, DecisionEngine


INTENTS = {
    "replace_card": "The customer wants a replacement for a lost or damaged card.",
    "track_transfer": "The customer wants to locate a transfer that has not arrived.",
    "update_address": "The customer wants to change their postal address.",
    "other": "The request has another purpose or is too unclear to classify.",
}


def classify_intent(engine: DecisionEngine, message: str) -> ChoiceAnswer:
    response = engine.system_one(
        state=message,
        questions={
            "intent": Choice(
                instructions="What is the customer's main requested action?",
                criteria=INTENTS,
            )
        },
    )
    answer = response.answers["intent"]
    assert isinstance(answer, ChoiceAnswer)
    return answer


def main() -> None:
    engine = DecisionEngine.from_pretrained()
    answer = classify_intent(engine, "My card snapped. How do I get another one?")
    print(answer.choice)
    print(answer.probabilities)


if __name__ == "__main__":
    main()
```

Reuse `engine` for subsequent messages. Replace the descriptions with your
application's actual intents and give overlapping categories clearer
boundaries. If a message can request several things, ask separate yes/no
questions for those actions instead of treating a single `Choice` as
multi-label classification.
