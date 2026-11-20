# specdecode

**[Status](STATUS.md) · [Roadmap](ROADMAP.md) · [Contributing](CONTRIBUTING.md) · [Issues](https://github.com/SMC17/specdecode/issues)**

Pieces for constraining and routing LLM decoding, in Zig 0.16:

- **A logit mask kernel** (`maskLogitsF32`): it applies a packed `u64` allow-bitset to an `f32`
  logit vector in place, setting disallowed entries to `-inf`. It is exported as a C ABI
  (`libspecdecode`), as WebAssembly (`specdecode.wasm`) with an ESM wrapper, and through Python
  `ctypes`. This is the most usable part.
- **A toy type-state pruner** (`SemanticPruner`) over a tiny call grammar,
  `name ( arg , arg )`. It checks known symbols, arity, argument types and numeric ranges, and
  can build an allow-mask over a vocabulary of whole-token strings.
- **A KV block bookkeeping model** (`KvPagePool`). It deduplicates blocks by Blake3 of their
  token bytes, pins "sink" blocks, and evicts by an attention/entropy score. It tracks metadata
  only: it holds no KV tensors.
- **A draft/frontier router** (`CascadeRouter`). It escalates a token when the pruner rejects it
  or when draft confidence is below 0.35, and also when a cumulative failure statistic
  crosses a threshold.

Status: **unit-tested** (6 Zig tests, plus the Node WASM and Python ctypes smoke tests). This is
a prototype. Nothing here is integrated into llama.cpp, MLX or vLLM. The C/WASM mask is the
piece that would plug into one.

## What it is not

- **The pruner is not a grammar engine for real tokenizers.** It works on whole-word string
  tokens of one call form. Real BPE vocabularies split identifiers across tokens, and that case
  is not handled.
- **The router's statistic is CUSUM, not an SPRT or an e-process.** It is clamped at zero, so the
  `alpha <= 0.05` in the code comment is not an error bound
  ([#3](https://github.com/SMC17/specdecode/issues/3)).

## Build

```sh
# Zig 0.16.0
zig build test --summary all            # 6/6
zig build -Doptimize=ReleaseFast        # also builds zig-out/lib/libspecdecode.{dylib,so}
zig build wasm                          # zig-out/bin/specdecode.wasm
node test_wasm.mjs                      # [ 1, -Infinity, 3, -Infinity ]
python3 python/specdecode.py            # ctypes smoke test
./zig-out/bin/specdecode bench
python3 benchmarks/bench_128k.py
```

C usage (`include/specdecode.h`). Allowed tokens have bit `i & 63` set in `mask_words[i >> 6]`:

```c
int32_t rc = specdecode_mask_logits_f32(logits, n_logits, mask_words, n_words);
```

JavaScript:

```js
import { createSpecdecode } from "./npm/index.mjs";
const sd = await createSpecdecode();
const logits = new Float32Array([1, 2, 3, 4]);
sd.maskLogits(logits, new BigUint64Array([0b0101n])); // -> [1, -Infinity, 3, -Infinity]
```

## Measured

Apple M2 Max, macOS 26.4, Zig 0.16.0 ReleaseFast, commit `e119167`, 2026-11-19.

| What | Workload | Result | Script |
| --- | --- | ---: | --- |
| `maskLogitsF32` | 128,256 logits, alternating-bit mask, 100 runs | 12–13 µs per mask | `specdecode bench` |
| Same through Python `ctypes` | 128,256 logits, 200 steps | 0.0195 ms per step | `benchmarks/bench_128k.py` |
| `buildLogitMask` | **8-token** vocabulary | ~85 ns | `specdecode bench` |
| `KvPagePool.admitBlock` | 2,000 admits, 64-block pool | ~300 ns | `specdecode bench` |
| `CascadeRouter.routeStep` | 10,000 steps, one fixed token | ~16 ns | `specdecode bench` |

The ~85 ns pruner figure covers 8 tokens. On a real vocabulary the pruner runs once per token
per step, so cost grows with vocabulary size, and that has not been measured. `bench_128k.py`
also prints a speedup over a pure-Python loop. That comparator is a strawman: NumPy masking
is the fair baseline, and it has not been measured yet.

## License

MIT
