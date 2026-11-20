import ctypes
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "python"))
from specdecode import load_lib  # noqa: E402

N_LOGITS = 128_256
N_STEPS = 200
N_WORDS = (N_LOGITS + 63) // 64
MASK_WORD = 0x5555555555555555


def main() -> None:
    lib = load_lib()

    c_logits = (ctypes.c_float * N_LOGITS)(*(float(i % 100) for i in range(N_LOGITS)))
    c_words = (ctypes.c_uint64 * N_WORDS)(*([MASK_WORD] * N_WORDS))

    # Warmup & correctness check
    rc = lib.specdecode_mask_logits_f32(c_logits, N_LOGITS, c_words, N_WORDS)
    assert rc == 0
    assert c_logits[0] == 0.0 and c_logits[1] == float("-inf")

    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        lib.specdecode_mask_logits_f32(c_logits, N_LOGITS, c_words, N_WORDS)
    c_elapsed_ms = (time.perf_counter() - t0) * 1000.0
    c_ms_per_step = c_elapsed_ms / N_STEPS

    py_logits = [float(i % 100) for i in range(N_LOGITS)]
    py_words = [MASK_WORD] * N_WORDS
    neg_inf = float("-inf")

    t1 = time.perf_counter()
    for _ in range(N_STEPS):
        for i in range(N_LOGITS):
            if not ((py_words[i >> 6] >> (i & 63)) & 1):
                py_logits[i] = neg_inf
    py_elapsed_ms = (time.perf_counter() - t1) * 1000.0
    py_ms_per_step = py_elapsed_ms / N_STEPS

    assert py_logits[0] == 0.0 and py_logits[1] == neg_inf

    print(f"vocab={N_LOGITS} steps={N_STEPS}")
    print(f"libspecdecode (ctypes): {c_ms_per_step:.4f} ms/step ({c_elapsed_ms:.2f} ms total)")
    print(f"pure python loop:       {py_ms_per_step:.4f} ms/step  (a strawman, kept for reference)")

    # The fair comparator: NumPy, unpacking the same packed bitset and masking in place.
    try:
        import numpy as np
    except ImportError:
        print("numpy:                  not installed (uv run --with numpy python3 benchmarks/bench_128k.py)")
        return
    np_logits = np.array([float(i % 100) for i in range(N_LOGITS)], dtype=np.float32)
    np_words = np.full(N_WORDS, MASK_WORD, dtype=np.uint64)
    keep = np.unpackbits(np_words.view(np.uint8), bitorder="little")[:N_LOGITS].astype(bool)
    np_logits[~keep] = -np.inf  # warm-up
    assert np_logits[0] == 0.0 and np_logits[1] == -np.inf
    t2 = time.perf_counter()
    for _ in range(N_STEPS):
        keep = np.unpackbits(np_words.view(np.uint8), bitorder="little")[:N_LOGITS].astype(bool)
        np_logits[~keep] = -np.inf
    np_ms_per_step = (time.perf_counter() - t2) * 1000.0 / N_STEPS
    print(f"numpy (unpack + mask):  {np_ms_per_step:.4f} ms/step")
    print(f"libspecdecode vs numpy: {np_ms_per_step / c_ms_per_step:.1f}x")


if __name__ == "__main__":
    main()
