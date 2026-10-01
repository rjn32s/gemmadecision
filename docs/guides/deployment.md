# Deploy a reusable model

Choose a persistent HTTP server when several applications share the model.
For a Python application running on one machine, keeping one warm local engine
is enough. Both approaches use the same pinned model and decision format.

## Download once, then run offline

```bash
python -m pip install 'gemmadecision[serve]'
gemmadecision download --output ./model
gemmadecision serve --model-path ./model --device cpu --offline
```

Copy the complete model directory when moving to another machine. The package
verifies the CPU ONNX graph and runtime files against its pinned manifest.
For Torch or vLLM, download native files with `--backend torch` or `--backend vllm`. It does not require a Hugging Face token for these public
weights. `--offline` avoids a Hub lookup when using the existing cache;
`--model-path` points directly to local files.

## CPU Docker image

The repository contains a CPU-only Dockerfile. Build it from the source tree:

```bash
git clone https://github.com/rjn32s/gemmadecision.git
cd gemmadecision
docker build -t gemmadecision:local .
```

Create a key in your shell, then pass it into the container:

```bash
export GEMMADECISION_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker run --rm \
  -p 127.0.0.1:8700:8700 \
  -e GEMMADECISION_API_KEY \
  -v gemmadecision-models:/models \
  gemmadecision:local
```

The named volume preserves downloaded weights across container restarts.
The process runs as user ID `10001`; a custom bind-mounted model cache must be
writable by that user. The container listens on all of its interfaces, while
the command above publishes the port only on your host's loopback interface.

Check `http://127.0.0.1:8700/ready` after the first download and load. Python
clients launched from the same shell pick up the key automatically. The image
uses CPU ONNX Runtime without Torch. GPU serving needs the Torch or vLLM extra
and a compatible GPU image/runtime. Adding `--gpus` does not change the
backend or install those dependencies.

## Run a function on Modal

This small example installs the published package in a remote container and
persists its model download in a Modal Volume. Inference runs on Modal; your
computer only starts the function and receives the result.

Install and authenticate the Modal launcher:

```bash
python -m pip install modal
modal setup
```

Save the following as `modal_app.py`:

```python
import modal

app = modal.App("gemmadecision-example")
model_cache = modal.Volume.from_name(
    "gemmadecision-model-cache", create_if_missing=True
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("gemmadecision==0.2.0")
    .env({"HF_HOME": "/cache/huggingface"})
)

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=300,
    volumes={"/cache": model_cache},
)
def route_ticket(text: str) -> str:
    from gemmadecision import decide

    result = decide(
        text,
        choices={
            "billing": "Charges, payments and refunds",
            "technical": "App crashes and sign-in failures",
            "review": "Requests outside those teams or needing clarification",
        },
        question="Which team should handle this request?",
    )
    model_cache.commit()
    return result

@app.local_entrypoint()
def main():
    print(route_ticket.remote("The same payment appears twice."))
```

Run it on a CPU:

```bash
modal run modal_app.py
```

This example uses CPU ONNX even if a GPU is attached. For GPU inference,
install `gemmadecision[torch]==0.2.0` in the image, request the desired GPU,
and use `DecisionEngine.from_pretrained(device="cuda")` inside the remote
function. Reuse that engine for later requests. See the
[serving guide](serving.md) for the equivalent explicit GPU server command.
Cloud compute and persistent storage follow your Modal plan's pricing. No
remote resources are created merely by reading this example.

The first container start still downloads and loads weights. The Volume saves
the download; a new container still loads the model into memory. Calls in a
reused container share the process-local engine. This example exposes a Modal
function, not a public HTTP endpoint. For an HTTP deployment, run the
[serving command](serving.md) in your chosen server environment.

The recipe follows Modal's [function entrypoints](https://modal.com/docs/guide/apps),
[GPU configuration](https://modal.com/docs/guide/gpu) and
[persistent Volumes](https://modal.com/docs/guide/volumes). It is a deployment
example, not an additional benchmark run.

## Optional vLLM serving

Use a separate Linux/CUDA environment:

```bash
python -m pip install 'gemmadecision[vllm]'
gemmadecision serve --backend vllm --device cuda
```

The [vLLM guide](../vllm.md) describes the pinned runtime, model export and
historical GPU measurements and score differences. The current CPU default
is ONNX; the recorded Torch/vLLM comparison does not measure it. vLLM support
does not imply that every workload will run faster.

For the historical 0.1.0 CPU/GPU installation checks, see
[installation validation](../installed-package-validation.md). For the exact
measured hardware and workloads, see [performance](../performance.md).
