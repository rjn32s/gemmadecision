# HTTP API

Install `pip install 'gemmadecision[serve]'`, then start `gemmadecision serve`
and use `http://127.0.0.1:8700`. The default runtime is CPU ONNX. The service exposes
decision and ranking endpoints backed by one resident model and a bounded
batching queue. It is not an OpenAI chat-completions endpoint.

## Routes

| Method | Path | Purpose | API key checked? |
|---|---|---|---|
| `POST` | `/v1/systemone` | Answer named choice, yes/no, and rubric questions. | Yes, when configured. |
| `POST` | `/v1/rank` | Rank supplied candidates. | Yes, when configured. |
| `GET` | `/health` | Readiness, model/backend metadata, and queue counters. | No. |
| `GET` | `/ready` | Same response and readiness check as `/health`. | No. |
| `GET` | `/v1/models` | Served model ID and `typed-decisions` task. | No. |
| `GET` | `/docs`, `/redoc`, `/openapi.json` | Generated interactive documentation and schema. | No. |

When `GEMMADECISION_API_KEY` is set, send
`Authorization: Bearer <your-key>` on inference requests. The CLI requires a key
for a non-loopback bind unless `--allow-unauthenticated` is explicitly supplied.
Health and schema endpoints remain public.

## `POST /v1/systemone`

```json
{
  "state": "The customer reports a duplicate charge and requests help today.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Choose the support team.",
      "criteria": {
        "billing": "Help with payments and duplicate charges.",
        "technical": "Help with product errors."
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "The customer explicitly requests help today."
    },
    "detail": {
      "type": "score",
      "instructions": "How detailed is this request?",
      "criteria": ["Too little detail", "Problem identified", "Problem and evidence supplied"]
    }
  }
}
```

| Request field | Type | Required | Meaning |
|---|---|---|---|
| `state` | JSON value | Yes | Text or structured material to evaluate. |
| `questions` | Object | Yes | 1–32 named question objects; their `type` selects the schema. |
| `model` | String | No | Served model ID or an accepted alias. |

`instructions` is required in each question and accepts a JSON value. Choice
criteria are a 2–64 entry mapping; score criteria are a 2–64 element ordered
list. Noul criteria are optional descriptions under `false`/`true`, with
`no`/`yes` accepted as input aliases. Do not provide both aliases for one outcome.
The normalized Noul answer uses `false` and `true`.

| Response field | Meaning |
|---|---|
| `answers` | Mapping with exactly the requested question names. |
| Choice answer | `type: "choice"`, `choice`, `probabilities`, `confidence`, `scores`. |
| Noul answer | `type: "noul"`, `noul` (weight of `true`), `probabilities`, `confidence`, `scores`. |
| Score answer | `type: "score"`, `score` (expected zero-based level), `probabilities`, `confidence`, `scores`. |
| `usage` | `input_tokens`, `output_tokens`, `total_tokens`, and `cached_pairs`. |
| `model` | Full served model ID including immutable revision. |
| `temperature` | Fixed value `4.136820402388508`. |
| `probabilities_source` | `softmax_ranking_scores_external_temperature`. |
| `latency_ms` | Server-side elapsed inference handling time, including queueing. |

Choice returns the largest score's label. Noul returns a float from 0 to 1,
not a JSON boolean. Score returns `sum(i * probabilities[str(i)])`, which can
lie between levels. All wire `confidence` fields are the largest probability,
including when the preferred Noul outcome is false.

## `POST /v1/rank`

```json
{
  "state": "I cannot sign in to my account.",
  "question": "Which team should handle this?",
  "candidates": {
    "account": "Help with passwords and account access.",
    "billing": "Help with payments and invoices."
  }
}
```

| Request field | Type | Default |
|---|---|---|
| `state` | JSON value; `context` is an accepted input alias. | Required. |
| `candidates` | Label-to-string mapping or list of strings; `answers` is an input alias. | Required, 2–64 candidates. |
| `question` | JSON value | Empty string. |
| `model` | String | Pinned served model ID. |

The response has `ranked`, a list ordered from highest to lowest raw score.
Each item contains `rank` (1-based), `candidate` (label), `text`, `score`, and
`probability`. List input receives string-index labels. Ties preserve input
order. The response also carries the same usage and probability-provenance
fields as `/v1/systemone`.

For example, save a request as `request.json` and submit it with:

```sh
curl --fail-with-body http://127.0.0.1:8700/v1/rank \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $GEMMADECISION_API_KEY" \
  --data-binary @request.json
```

## Model identity

The default ID is
`rajan2k/GemmaDecision-270M@785d530221c990671f29976902540101bb9c7647`.
Accepted aliases are `rajan2k/GemmaDecision-270M`, `GemmaDecision-270M`, and
`gemmadecision`. Supplying another model ID fails validation; it does not
download a different model. `/v1/models` returns a `models` list, not an OpenAI
`data` response envelope.

## Status codes and operational limits

| Status | Meaning |
|---|---|
| `200` | Successful inference or ready health response. |
| `401` | Missing or incorrect bearer key on an inference route. |
| `413` | POST body exceeds 2 MiB (2,097,152 bytes). |
| `422` | Invalid JSON/schema, unsupported model, invalid candidates, or token/input limits exceeded. |
| `500` | Inference backend failure or unavailable inference worker. |
| `503` | Full inference queue, or failed readiness check. Queue-full responses include `Retry-After: 1`. |
| `504` | Request exceeded the configured server deadline. No automatic retry is performed. |

Errors use a `detail` field; schema-validation errors may contain a list of
field errors rather than a string. Unknown fields are rejected. Request-size
checking applies to POST bodies before inference. It does not extend model
token limits: rendered state/question is capped at 2,048 tokens and each
candidate at 768, including special tokens. Across all questions, a request
may contain at most 256 candidate pairs. Text is never silently truncated.

Default server timeout is 60 seconds, queue capacity is 128, and a scheduler
batch contains up to 32 requests. A timed-out request may have already started
model work; returning a timeout cannot undo a running tensor operation.

The health counters `completed_requests` and `model_batches` describe successful
requests and scheduler batch calls. A scheduler call can contain several encoder
forwards. Neither counter is a billed-token meter. More detail is in the
[CLI](cli.md) and [limits](limits.md) references.
