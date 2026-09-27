# Route a support ticket

Choose the team that should read a ticket. Give each label a description so
the decision depends on the team's responsibility, not an opaque queue name.

```python
from gemmadecision import decide


TEAMS = {
    "billing": "Charges, invoices, refunds and subscription payments.",
    "account": "Signing in, account access and profile changes.",
    "technical": "Software errors, broken features and installation problems.",
    "review": "The request is unclear or does not fit the other teams.",
}


def route_ticket(ticket: str) -> str:
    return decide(
        state=ticket,
        choices=TEAMS,
        question="Which team should handle this support ticket?",
    )


def main() -> None:
    ticket = "The invoice lists two payments for one monthly subscription."
    print(route_ticket(ticket))


if __name__ == "__main__":
    main()
```

The return value is one of the four dictionary keys. `decide()` loads a shared
engine on first use, so repeated calls in the same process reuse the model.

The `review` option gives unclear requests a possible destination. It is still
a model prediction; for an additional review rule based on scores, use the
[human-review recipe](human-review.md). This example returns a label, leaving
the actual queue update to your application.
