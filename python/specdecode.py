import ctypes
import pathlib
import sys


def load_lib(path: str | None = None) -> ctypes.CDLL:
    if path is None:
        ext = "dylib" if sys.platform == "darwin" else "so"
        path = str(pathlib.Path(__file__).resolve().parent.parent / "zig-out" / "lib" / f"libspecdecode.{ext}")
    lib = ctypes.CDLL(path)
    lib.specdecode_mask_logits_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    lib.specdecode_mask_logits_f32.restype = ctypes.c_int32
    return lib


if __name__ == "__main__":
    lib = load_lib()
    logits = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    words = (ctypes.c_uint64 * 1)(0b1010)
    rc = lib.specdecode_mask_logits_f32(logits, 4, words, 1)
    assert rc == 0
    assert logits[0] == float("-inf") and logits[1] == 2.0 and logits[2] == float("-inf") and logits[3] == 4.0
    print("python specdecode ctypes ok:", [logits[i] for i in range(4)])
