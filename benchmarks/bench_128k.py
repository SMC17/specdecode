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

    speedup = py_ms_per_step / c_ms_per_step
    print(f"vocab={N_LOGITS} steps={N_STEPS}")
    print(f"libspecdecode (ctypes): {c_ms_per_step:.4f} ms/step ({c_elapsed_ms:.2f} ms total)")
    print(f"pure python loop:       {py_ms_per_step:.4f} ms/step ({py_elapsed_ms:.2f} ms total)")
    print(f"speedup:                {speedup:.1f}x")


if __name__ == "__main__":
    main()
