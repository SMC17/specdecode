# specdecode

**Zero-GC Mid-Decode Semantic & Type-State Pruner, 128k-Vocab Logit-Mask C ABI (`libspecdecode`), Entropy-Guided KV-Cache Paging, and Speculative Cascade Router.**

## Capabilities
* **Mid-Decode Type-State & Contract Pruning:** Enforces symbol existence, function arity, argument types, and numeric bounds per decode step in nanoseconds.
* **128k-Vocab `f32` Logit Masking (`specdecode_mask_logits_f32`):** Packed `u64` bitmasking over 128,256+ logits in `~70 µs` for `llama.cpp`, `MLX`, and `vLLM`.
* **Entropy-Guided KV-Cache Paging & Prefix Dedup:** Blake3 prefix deduplication + attention-sink pinning + attention/entropy utility eviction.
* **Self-Verifying Cascade Router:** Routes tokens between local draft and frontier verifier using live semantic checks and an anytime-valid Bernoulli SPRT e-process.

## Quickstart
```sh
zig build test
zig build -Doptimize=ReleaseFast
./zig-out/bin/specdecode bench
```
