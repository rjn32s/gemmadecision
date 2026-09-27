# Score against an urgency rubric

Use `Score` for a small ordered set of described levels. Put the descriptions
in order, starting at level zero.

```python
from gemmadecision import DecisionEngine, Score, ScoreAnswer


LEVELS = [
    "Routine: the request is informational or can wait in the normal queue.",
    "Soon: the issue interferes with work, but a workaround is available.",
    "Now: the customer cannot continue their work and needs immediate assistance.",
]


def score_urgency(engine: DecisionEngine, message: str) -> ScoreAnswer:
    response = engine.system_one(
        state=message,
        questions={
            "urgency": Score(
                instructions="Which urgency level is supported by this request?",
                criteria=LEVELS,
            )
        },
    )
    answer = response.answers["urgency"]
    assert isinstance(answer, ScoreAnswer)
    return answer


def main() -> None:
    engine = DecisionEngine.from_pretrained()
    answer = score_urgency(
        engine,
        "The desktop app crashes, but I can finish the task in the browser for now.",
    )
    most_likely = max(answer.probabilities, key=answer.probabilities.get)
    print({
        "expected_level": answer.score,
        "most_likely_level": int(most_likely),
        "description": LEVELS[int(most_likely)],
        "distribution": answer.probabilities,
    })


if __name__ == "__main__":
    main()
```

`.score` is the probability-weighted mean of level indices, so it can be
fractional. The most likely level is the index with the largest probability;
it can differ from rounding the mean. Keep both concepts distinct when using
the result in a queue or dashboard. The levels express your rubric, not a
measured waiting time.
