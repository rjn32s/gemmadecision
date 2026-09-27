# Rank supplied responses

Choose among answers your application already has: approved templates,
retrieved suggestions, or candidates from another model. Put any facts the
answer should respect beside the user's request.

```python
from gemmadecision import rank


RESPONSES = {
    "self_service": "Open Settings, then Billing, then Download invoice for the selected month.",
    "contact_support": "Contact support and ask them to locate the invoice for you.",
    "unavailable": "Invoices cannot be downloaded from this product.",
}


def rank_responses(request: str, facts: str):
    return rank(
        state={"request": request, "product_facts": facts},
        choices=RESPONSES,
        question="Which response most directly answers the request while following the product facts?",
    )


def main() -> None:
    result = rank_responses(
        request="Where can I get last month's invoice?",
        facts="Customers can download monthly invoices from Settings > Billing.",
    )
    for item in result.ranked:
        print(item.rank, item.candidate, item.score, item.text)


if __name__ == "__main__":
    main()
```

`ranked` is ordered from highest to lowest score. `.candidate` is the key you
supplied; `.text` is its response text. Scores can be negative. Their ordering
matters within the request, and a larger score means preferred by this model.

The model does not generate or repair the winning response. Apply any required
content checks to your candidate pool. Probabilities are normalized across
that pool, so they change if you add or remove responses.
