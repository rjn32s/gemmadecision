# Label sentiment

Use named alternatives to describe what each sentiment means in your domain.
This example includes a mixed label so a review can contain both praise and
criticism without being forced into a single polarity.

```python
from gemmadecision import Choice, ChoiceAnswer, DecisionEngine


SENTIMENTS = {
    "positive": "The review expresses satisfaction or praise overall.",
    "negative": "The review expresses dissatisfaction or criticism overall.",
    "neutral": "The review is factual or has no clear positive or negative opinion.",
    "mixed": "The review expresses substantial praise and substantial criticism.",
}


def label_sentiment(engine: DecisionEngine, review: str) -> ChoiceAnswer:
    response = engine.system_one(
        state=review,
        questions={
            "sentiment": Choice(
                instructions="Which label best describes the reviewer's opinion of the product?",
                criteria=SENTIMENTS,
            )
        },
    )
    answer = response.answers["sentiment"]
    assert isinstance(answer, ChoiceAnswer)
    return answer


def main() -> None:
    engine = DecisionEngine.from_pretrained()
    review = "The setup guide was clear, but searching old files is still frustrating."
    answer = label_sentiment(engine, review)
    print(answer.choice)
    print(answer.probabilities)


if __name__ == "__main__":
    main()
```

For aspect-based sentiment, ask a separate question about each aspect, such as
setup and search, in one `system_one()` call. Keep each question explicit about
which aspect it concerns. The model is labeling the supplied text; it does
not know the reviewer's private intent.
