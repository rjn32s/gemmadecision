# Serve decisions over HTTP

Run one model process and share it across your application. The server uses
Granian for Rust HTTP serving and batched PyTorch for inference by default.

## Start the server

Install the package in the environment that will run the server:

```bash
python -m pip install gemmadecision
gemmadecision serve
```

The address is `http://127.0.0.1:8700`. The first start downloads the pinned
model, then loads it. Later starts reuse the downloaded files. Keep this
terminal open and use another terminal for requests.

Choose hardware explicitly when needed:

```bash
gemmadecision serve --device cpu
```

```bash
gemmadecision serve --device cuda
```

On a supported Apple Silicon installation, use `--device mps`. Without a
device argument, the package selects CUDA, then MPS, then CPU according to
availability. CUDA requires a compatible NVIDIA driver and PyTorch build.

Check readiness after loading:

```bash
curl http://127.0.0.1:8700/ready
```

Open `http://127.0.0.1:8700/docs` for interactive request schemas.

## Choose an option with curl

`POST /v1/systemone` accepts a state and one or more named questions. A choice
question maps the labels your application uses to descriptions the model sees.

```bash
curl http://127.0.0.1:8700/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "The same card payment appears twice on my statement.",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which support team should handle this request?",
        "criteria": {
          "billing": "Investigate charges, duplicate payments and refunds.",
          "technical": "Investigate app crashes and login failures."
        }
      }
    }
  }'
```

Read the selected label from `answers.team.choice`. That answer also includes
raw scores, derived probabilities and `confidence`, the largest derived
probability. These numbers need validation on your application's data.

For an ordered list of options, use `POST /v1/rank`:

```bash
curl http://127.0.0.1:8700/v1/rank \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "The same card payment appears twice on my statement.",
    "question": "Which action best addresses this request?",
    "candidates": {
      "check_payment": "Investigate the transaction records for a duplicate charge.",
      "reset_password": "Help the customer reset their account password."
    }
  }'
```

The `ranked` array is sorted from highest to lowest score. Mapping keys appear
as `candidate`; descriptions appear as `text`.

## Call the server from Python

Reuse a client instead of constructing one for every request:

```python
from gemmadecision import DecisionClient

with DecisionClient() as client:
    answer = client.decide(
        "The same payment appears twice.",
        candidates={
            "billing": "Charges, duplicate payments and refunds",
            "technical": "App crashes and login failures",
        },
        question="Which team should handle this request?",
    )
    print(answer.choice)
    print(answer.probabilities)
```

For an async application:

```python
import asyncio
from gemmadecision import AsyncDecisionClient

async def main():
    async with AsyncDecisionClient() as client:
        result = await client.rank(
            "The app closes whenever I try to sign in.",
            candidates={
                "technical": "Investigate app crashes and sign-in problems.",
                "billing": "Investigate payments and invoices.",
            },
        )
        print(result.ranked[0].candidate)

asyncio.run(main())
```

Both clients accept `base_url`, `api_key` and `timeout`. They reuse HTTP
connections and do not retry decisions automatically. For native PydanticAI
clients, see the [PydanticAI guide](../pydantic_ai.md).

## Connect from another machine

Set a key in the server shell and bind to the network interface:

```bash
export GEMMADECISION_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
gemmadecision serve --host 0.0.0.0 --port 8700
```

Provide the same key to your client through its environment or secret manager.
`DecisionClient` and `AsyncDecisionClient` read `GEMMADECISION_API_KEY` and
`GEMMADECISION_BASE_URL` automatically. Set the base URL to the server's actual
hostname or address; `0.0.0.0` is a bind address.

Requests made with curl include the key as a bearer header:

```bash
curl "$GEMMADECISION_BASE_URL/v1/rank" \
  -H "Authorization: Bearer $GEMMADECISION_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"state":"I cannot sign in.","candidates":["Account access support","Payment support"]}'
```

Use a TLS reverse proxy for an internet-facing service. The CLI requires a key
when binding outside loopback unless you deliberately pass
`--allow-unauthenticated`. `/health`, `/ready`, `/v1/models` and the API schema
remain readable without an inference key.

## Endpoints and tuning

| Endpoint | Purpose |
|---|---|
| `POST /v1/systemone` | Named choice, yes/no and rubric questions |
| `POST /v1/rank` | Ordered candidates with raw scores and derived probabilities |
| `GET /v1/models` | The pinned model served by this process |
| `GET /health`, `GET /ready` | Readiness and queue information |
| `GET /docs` | Interactive API documentation |

Start with the default batching settings. `--max-batch-size` limits candidate
pairs per Torch forward, while `--batch-requests` limits requests grouped by
the server scheduler. `--max-batch-tokens` bounds padded tokens in a Torch
batch. The score cache is off unless you set `--cache-size`.

See [measured performance](../performance.md) before choosing concurrency,
[deployment](deployment.md) for Docker and Modal, and
[troubleshooting](troubleshooting.md) for failures and input limits.
