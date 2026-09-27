# GemmaDecision on vLLM

The optional backend targets **vLLM 0.30.0 on Linux with a BF16-capable CUDA
GPU**. Its CPU fallback is the default PyTorch backend. vLLM has a substantial
installation and startup cost; for a 270M model and one request at a time it is
not automatically faster. Choose a backend using the measured latency for
your own candidate counts and request concurrency.

The vLLM backend is **experimental until GPU parity and performance have been
measured for your deployment**. The unit tests validate packaging and the
pooling contract; they do not prove GPU kernel equivalence.

```bash
pip install 'gemmadecision[vllm]'
gemmadecision serve --backend vllm
```

The package downloads the pinned GemmaDecision v4 model, creates an
encoder-only view in its cache, and serves the usual ranking API. Use a single
server worker per GPU: multiple workers instantiate independent model engines.
vLLM 0.30.0 pins PyTorch 2.13.0; install this extra in its own environment rather
than forcing another CUDA PyTorch version into that environment.

For direct backend use with a complete local model:

```python
from gemmadecision.backends.vllm import VLLMBackend

if __name__ == '__main__':
    backend = VLLMBackend('/path/to/GemmaDecision-270M')
    scores = backend.score_pairs([
        'The card payment was taken twice.\n\nChoose a support team.'
        '\n\nCandidate action:\nBilling and refunds',
        'The card payment was taken twice.\n\nChoose a support team.'
        '\n\nCandidate action:\nLogin and password support',
    ])
    print(scores)  # Raw ranking scores, not probabilities.
```

Prefer the public Engine/client API for normal use; it validates candidate and
state limits and constructs the exact trained text. `score_pairs` is a lower
level interface for already-rendered joint texts.

## How this follows the CLM serving pattern

[CLM's official recipe](https://github.com/Contrastive-LM/CLM#quickstart) runs a
vLLM pooling encoder and applies its learned head in a separate API layer.
GemmaDecision uses the same separation of responsibilities, with an important
model-specific requirement: **each whole state + question + candidate is one
encoder input**. Candidate-only embeddings cannot be reused across different
states with this trained model. CLM's independently cached state and action
representations belong to a different architecture.

The implementation uses vLLM's
[pooling conversion](https://docs.vllm.ai/en/v0.30.0/models/pooling_models/):

1. Keep every encoder tensor unchanged and change only the exported
   `architectures` metadata to `Gemma3ForCausalLM`.
2. Run `runner='pooling'`, `convert='embed'`, `pooling_type='LAST'`, and
   `use_activation=False`.
3. Send explicit tokenizer IDs, including the frozen tokenizer's special
   tokens, with no chat template or truncation.
4. Convert raw last-token hidden states to FP32, L2-normalize them, and apply
   the original LayerNorm → Linear → GELU → Linear scalar head in FP32.

The small head runs on CPU and receives only 640 values per candidate. The
decoder runs on CUDA. This avoids coupling the package to internal vLLM worker
classes. Scores can differ slightly from the reference because GPU kernels and
batch shapes differ; near-tied candidates can change order. No quantization is
enabled. The package does not apply an unvalidated probability calibration.

vLLM's
[pooling weight adapter](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/adapters.py)
accepts the bare encoder tensor names by mapping them under `model.`. No tensor
rewrite, random language-model head, or model fine-tuning is required.

## Standalone encoder export

```bash
python -m gemmadecision.export_vllm \
  --model /path/to/GemmaDecision-270M --output ./gemmadecision-encoder
vllm serve ./gemmadecision-encoder \
  --runner pooling --convert embed --dtype bfloat16 \
  --pooler-config '{"pooling_type":"LAST","use_activation":false}' \
  --max-model-len 3072 --gpu-memory-utilization 0.20 --port 8090
```

This standalone endpoint returns **encoder vectors only**, not GemmaDecision
rankings. The package's `serve --backend vllm` route additionally applies the
trained scalar head. The export intentionally omits `joint_head.safetensors`:
vLLM's generic loader scans safetensor files and must not mistake that head for
decoder weights. The export includes hashes, tokenizer files, and the original
license documents. Existing exports are verified before reuse.

The default backend disables prefix caching and uses eager execution for a
predictable first deployment. `VLLMBackend(enforce_eager=False)` can enable
vLLM's graph/compilation path, but benchmark cold start, warm p50/p95 latency,
throughput, memory, and score parity before adopting it. Rust serves HTTP
connections; vLLM/CUDA performs the neural-network computation.
