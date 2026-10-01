# CLI and environment

The normal package install includes the CPU ONNX runtime and the
`gemmadecision` command. Add the `serve` extra for the HTTP server:

```sh
pip install 'gemmadecision[serve]'
gemmadecision --version
gemmadecision doctor
gemmadecision serve
```

`serve` loads one resident model and starts the Rust HTTP runtime on
`127.0.0.1:8700`. First use downloads the pinned public weights if absent from
the Hugging Face cache. No Hugging Face token is required. The default does
not install or import Torch or Transformers.

## Commands

| Command | Action |
|---|---|
| `gemmadecision --version` | Print package version. |
| `gemmadecision doctor` | Print Python and relevant installed package versions; does not load a model or test GPU availability. |
| `gemmadecision download [--output DIRECTORY] [--backend onnx\|torch\|vllm]` | Download pinned files for the selected runtime (default `onnx`) and print their directory. Without `--output`, use the Hugging Face cache. |
| `gemmadecision serve [OPTIONS]` | Load the model and start HTTP serving. |
| `gemmadecision serve --help` | List all serving flags. |

## Serving options

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. Non-loopback requires an API key or explicit opt-in to no authentication. |
| `--port` | `8700` | TCP port, 1–65535. |
| `--model-path` | Unset | Load a complete local model directory instead of downloading. |
| `--backend` | `auto` | `auto`, `onnx`, `torch`, or `vllm`. Auto uses ONNX on CPU; explicit CUDA/MPS or a legacy native model directory selects Torch. |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps`. ONNX supports CPU only; within an explicitly selected Torch backend, auto chooses CUDA, MPS, then CPU. vLLM requires auto or CUDA. |
| `--strict` | Off | ONNX/Torch singleton mode; only Torch preserves the native reference execution shape. Rejected with vLLM. |
| `--offline` | Off | If no local path is supplied, require the pinned snapshot to exist in the HF cache. |
| `--max-batch-tokens` | `8192` | ONNX/Torch padded-token budget per forward; positive integer. |
| `--max-batch-size` | `32` | ONNX/Torch candidate pairs per forward; vLLM concurrent-sequence limit. |
| `--batch-wait-ms` | `1.0` | Scheduler gathering window, between 0 and 100 ms. |
| `--batch-requests` | `32` | Maximum HTTP requests in one scheduler batch; positive integer. |
| `--queue-size` | `128` | Inference queue capacity and Granian backpressure setting; positive integer. |
| `--request-timeout` | `60` | Server deadline in seconds for queued preparation/inference; positive value. |
| `--cache-size` | `0` | Exact joint-input score-cache entries. Zero disables it; maximum 1,000,000. |
| `--gpu-memory-utilization` | `0.2` | vLLM GPU memory fraction, strictly between 0 and 1; ignored by ONNX/Torch. |
| `--allow-unauthenticated` | Off | Allow a non-loopback bind without an API key. Does not bypass a configured key. |

`--max-batch-size` counts candidate pairs, whereas `--batch-requests` counts
HTTP requests. For example, eight four-candidate requests can produce 32 joint
inputs. Length bucketing and the padded-token budget may split them into
several encoder forwards.

The server uses one Granian worker/model process. There is no `--workers` flag
in this CLI. Adding external worker processes would load separate model copies.

## CPU or CUDA

```sh
gemmadecision serve --device cpu
```

On a host with a compatible CUDA installation:

```sh
pip install 'gemmadecision[serve,torch]'
gemmadecision serve --device cuda --max-batch-size 32 --max-batch-tokens 8192
```

See the [measured CPU results](../cpu-latency.md) and
[GPU measurements](../performance.md) before choosing batch settings.

## Offline startup

```sh
gemmadecision download --output ./models/gemmadecision
gemmadecision serve --model-path ./models/gemmadecision --offline
```

The default download contains the ONNX graph, trusted manifest, configuration,
and tokenizer files. Loading verifies their expected hashes. Native Torch/vLLM
needs its corresponding files:

```sh
gemmadecision download --backend torch --output ./models/gemmadecision-torch
```

Select `--backend torch` and that directory when serving.
Copying only the original Google base model will not provide the tuned model.

## Network serving

Set `GEMMADECISION_API_KEY` through your shell or secret manager, then run:

```sh
gemmadecision serve --host 0.0.0.0 --port 8700
```

Clients send `Authorization: Bearer <key>`. Health, model-list, and schema
endpoints do not require the key. The CLI does not configure TLS; provide it
through the deployment platform or reverse proxy when needed.

## vLLM

Install the optional backend on a supported Linux CUDA host:

```sh
pip install 'gemmadecision[vllm]'
gemmadecision serve --backend vllm --device cuda --gpu-memory-utilization 0.2
```

Version 0.2.0 pins vLLM to 0.30.0. The package performs the model export and uses
vLLM's pooling runner plus the trained scalar head; it does not call a chat
generation endpoint. The vLLM backend retains the state/candidate limits and
uses its own sequence scheduling. `--max-batch-tokens` is an ONNX/Torch batching
control, not a vLLM scheduler setting. See [vLLM deployment](../vllm.md).

## Environment variables

| Variable | Scope | Meaning |
|---|---|---|
| `GEMMADECISION_API_KEY` | Server and HTTP clients | Bearer key. Explicit client `api_key` takes precedence. |
| `GEMMADECISION_BASE_URL` | `DecisionClient` / `AsyncDecisionClient` | Default server URL when `base_url` is not provided. |
| `GEMMADECISION_MODEL_PATH` | Simple local `decide`/`rank` and default local PydanticAI model | Directory used on the first lazy engine load. The serving CLI uses `--model-path` instead. |
| `HF_HOME` | Hugging Face downloader | Cache root; set before downloading/loading. |
| `HF_HUB_OFFLINE` | Hugging Face runtime | Disable Hub network access; a cached/local model is still required. |
| `TOKENIZERS_PARALLELISM` | Tokenizer runtime | The serving CLI defaults it to `false` if unset. |
| `HF_HUB_DISABLE_TELEMETRY` | Hub runtime | The serving CLI defaults it to `1` if unset. |
| `GEMMADECISION_CONFIG` | Advanced ASGI embedding | JSON server configuration read by `gemmadecision.server`; the CLI overwrites it with parsed flags. |

`GEMMADECISION_CONFIG` accepts the engine options in the Python reference plus
`queue_size`, `batch_requests`, `batch_wait_ms`, and `request_timeout`. Prefer CLI
flags unless embedding the ASGI app deliberately. Setting an environment
variable after a shared engine is already loaded does not reconfigure that
engine.
