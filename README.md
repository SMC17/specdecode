# specdecode

**Zero-GC Mid-Decode Semantic & Type-State Pruner, 128k-Vocab Logit-Mask C ABI (`libspecdecode`), WebAssembly SIMD (`specdecode.wasm`), Entropy-Guided KV-Cache Paging, and Speculative Cascade Router.**

## Capabilities
* **Mid-Decode Type-State & Contract Pruning:** Enforces symbol existence, function arity, argument types, and numeric bounds per decode step in `87 ns`.
* **128k-Vocab `f32` Logit Masking (`specdecode_mask_logits_f32`):** Portable SIMD packed `u64` bitmasking over 128,256+ logits in `16 µs` (`0.0177 ms/step` via Python `ctypes`) for `llama.cpp`, `MLX`, `vLLM`, Node, Edge, and Browsers.
* **Entropy-Guided KV-Cache Paging & Prefix Dedup:** Blake3 prefix deduplication + attention-sink pinning + attention/entropy utility eviction (`275 ns/admit`).
* **Self-Verifying Cascade Router:** Routes tokens between local draft and frontier verifier using live semantic checks and an anytime-valid Bernoulli SPRT e-process (`16 ns/route`).

## Build & Test
```sh
zig build test
zig build -Doptimize=ReleaseFast
zig build wasm
./zig-out/bin/specdecode bench
```

## Benchmarks (128,256 `f32` Logits, 200 Decode Steps)
Measured on Apple Silicon with `zig build -Doptimize=ReleaseFast`:
- Native Zig (`./zig-out/bin/specdecode bench`): `16 µs` per 128k mask (`87 ns` prune step, `275 ns` KV admit, `16 ns` cascade route)
- Python `ctypes` (`python3 benchmarks/bench_128k.py`): `0.0177 ms/step` (`3.55 ms` total) vs pure Python loop `10.7886 ms/step` (`2157.72 ms` total) — **`607.8x` speedup**

## C / C++ (`include/specdecode.h`)
Link against `zig-out/lib/libspecdecode.dylib` (macOS) or `zig-out/lib/libspecdecode.so` (Linux):
```c
#include "include/specdecode.h"

// Allowed tokens have bit (i & 63) set to 1 in mask_words[i >> 6].
// Disallowed logits are overwritten with -INFINITY in-place.
int32_t rc = specdecode_mask_logits_f32(logits, n_logits, mask_words, n_words);
```

## Python (`python/specdecode.py`)
```python
import ctypes
from python.specdecode import load_lib

lib = load_lib()
logits = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
words = (ctypes.c_uint64 * 1)(0b1010)  # allow indices 1 and 3
assert lib.specdecode_mask_logits_f32(logits, 4, words, 1) == 0
```
Run smoke test and benchmark:
```sh
python3 python/specdecode.py
python3 benchmarks/bench_128k.py
```

## npm / WebAssembly (`npm/index.mjs`)
Zero-dependency ESM + TypeScript wrapper around `specdecode.wasm` for Node, Edge runtimes, and Browsers:
```js
import { createSpecdecode } from "./npm/index.mjs";

const sd = await createSpecdecode();
const logits = new Float32Array([1.0, 2.0, 3.0, 4.0]);
const words = new BigUint64Array([0b0101n]); // allow indices 0 and 2
sd.maskLogits(logits, words);
// logits is now Float32Array [1, -Infinity, 3, -Infinity]
```
Run the Node WASM test:
```sh
node test_wasm.mjs
```
