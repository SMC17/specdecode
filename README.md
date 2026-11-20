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
  or when draft confidence is below 0.35. It also escalates when a CUSUM e-detector over those
  failures crosses its threshold. With failures at the expected 5% rate, the mean steps
  between drift escalations is at least the threshold (default 1,000).

Status: **unit-tested** (7 Zig tests, plus the Node WASM and Python ctypes smoke tests). This is
a prototype. Nothing here is integrated into llama.cpp, MLX or vLLM. The C/WASM mask is the
piece that would plug into one.

## What it is not

- **The pruner is not a grammar engine for real tokenizers.** It works on whole-word string
  tokens of one call form. Real BPE vocabularies split identifiers across tokens, and that case
  is not handled.
- **The pruner is slow on real vocabularies.** It runs `stepToken` once per vocabulary entry per
  step: ~0.42–0.47 ms per step on 32,000 tokens (see below). That fits a 50 token/s budget, but
  not a fast local draft model.
- **The router's threshold is a run-length bound, not an error rate.** It assumes independent
  failures. Before [#3](https://github.com/SMC17/specdecode/issues/3), the code comment
  claimed `alpha <= 0.05` at a threshold of 20.

## Build

```sh
# Zig 0.16.0
zig build test --summary all            # 7/7
zig build -Doptimize=ReleaseFast        # also builds zig-out/lib/libspecdecode.{dylib,so}
zig build wasm                          # zig-out/bin/specdecode.wasm
node test_wasm.mjs                      # [ 1, -Infinity, 3, -Infinity ]
python3 python/specdecode.py            # ctypes smoke test
./zig-out/bin/specdecode bench
uv run --with numpy python3 benchmarks/bench_128k.py   # adds the NumPy comparison
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

Apple M2 Max, macOS 26.4, Zig 0.16.0 ReleaseFast, 2026-11-19, three runs each.

| What | Workload | Result | Script |
| --- | --- | ---: | --- |
| `maskLogitsF32` | 128,256 logits, alternating-bit mask, 100 runs | 12–13 µs per mask | `specdecode bench` |
| Same through Python `ctypes` | 128,256 logits, 200 steps | 0.0177 ms per step | `benchmarks/bench_128k.py` |
| NumPy (unpack bitset + boolean mask) | same | 0.25–0.26 ms per step (**~14×** slower than ctypes) | same, with `uv run --with numpy` |
| `buildLogitMask` | **32,000**-token vocabulary, 200 steps | **419–467 µs** per step | `specdecode bench` |
| `buildLogitMask` | 8-token vocabulary | 82–88 ns per step | `specdecode bench` |
| `KvPagePool.admitBlock` | 2,000 admits, 64-block pool | ~300 ns | `specdecode bench` |
| `CascadeRouter.routeStep` | 10,000 steps, one fixed token | ~16 ns | `specdecode bench` |

Earlier versions of this README headlined "87 ns per prune step". That was the 8-token case:
the cost of a real vocabulary is about 5,000 times higher. They also headlined a "607×"
speedup, which was measured against a pure-Python loop. Against NumPy, the fair baseline, the
speedup is about 14×.

## License

MIT
