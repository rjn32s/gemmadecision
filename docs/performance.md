# Performance and numerical behavior

The PyTorch backend loads the **fully tuned v0.4.0 Gemma encoder and scalar
head**, not the original Google weights. The model stays resident between
requests. A Rust HTTP runtime handles connections; transformer tensor kernels
remain in PyTorch/SDPA. Rust alone does not make model math faster.

The inference backend groups joint inputs by length, pads on the right, and
bounds each batch by both input count and **padded** token count. The scheduler
can combine candidate pairs from several concurrent requests in one forward
pass. Each output is restored to its original request and candidate position.
There is no generation loop, vocabulary projection, KV decoding cache, or chat
template. The encoder uses its final nonpadding token; L2 normalization and the
learned scalar head run in FP32. CUDA uses a BF16 encoder where supported;
CPU/MPS use FP32, matching the published reference's dtype policy.

State and candidate cannot be cached as independent embeddings: this model
encodes their concatenation jointly. Such a cache would change the model.

## Controls

- `max_batch_size` bounds the number of joint inputs per encoder forward pass.
- `max_batch_tokens` bounds the batch's padded token count. The default is
  8,192; a single longer joint input is rejected instead of silently truncated.
- `strict=True` uses one input per forward pass, preserving the frozen
  reference's numerical batch shape. It is useful for parity checks.
- Batching trades a small configurable queue delay for higher throughput. A
  concurrent throughput result is not the same metric as single-request latency.

The published state/question limit of 2,048 tokens and per-candidate limit of
768 tokens include tokenizer special tokens. They are checked separately before
scoring. The model's longer architectural context is not a justification for
silently extending its validated input contract. No text is truncated.

GPU kernels can change floating-point rounding across batch shapes. This can
alter close ties, even though the ranking architecture has no cross-candidate
interaction. Report maximum logit difference and ranking agreement against the
frozen singleton implementation on named inputs before claiming parity.

## Measuring a release

A useful report names the exact model revision, software versions, device,
dtype, candidate count, token lengths, padding policy, batching controls,
concurrency, and warmup. Report warm p50/p95 latency, requests/second and
candidate pairs/second; report cold download/load time separately. Compare
singleton and batched implementations on the **same** inputs and hardware.
Report model precision and actual memory use. Never infer CPU/Mac performance
from an H100 benchmark, or attribute CUDA kernel work to Rust.

Unit tests use tiny synthetic tensors and test batch bounds, ordering, pooling,
normalization, failure behavior, and file integrity. They do not establish
full-model speed or floating-point parity. Real-model measurements belong in
the release's measured-results artifact; no speedup is promised by this document.
