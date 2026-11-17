export type WasmSource =
  | BufferSource
  | WebAssembly.Module
  | Response
  | URL
  | string;

export interface SpecdecodeWasmExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  wasm_mask_logits_f32(
    logitsPtr: number,
    nLogits: number,
    wordsPtr: number,
    nWords: number,
  ): void;
}

export interface SpecdecodeInstance {
  readonly memory: WebAssembly.Memory;
  readonly exports: SpecdecodeWasmExports;

  /**
   * Ensure linear WebAssembly memory has at least `minBytes` bytes, growing if needed.
   */
  ensureMemory(minBytes: number): void;

  /**
   * Mask `logits` in-place using packed 64-bit `maskWords` (LSB-first per word).
   * Disallowed indices are set to `-Infinity`. Returns `logits`.
   */
  maskLogits(logits: Float32Array, maskWords: BigUint64Array): Float32Array;

  /**
   * Zero-copy call directly against existing offsets inside `this.memory.buffer`.
   */
  maskLogitsInPlace(
    logitsPtr: number,
    nLogits: number,
    wordsPtr: number,
    nWords: number,
  ): void;
}

/**
 * Instantiate `specdecode.wasm` for Node, Edge runtimes, or Browsers.
 * When `source` is omitted, loads `specdecode.wasm` adjacent to this module
 * (or `../zig-out/bin/specdecode.wasm` in Node when running from source).
 */
export function createSpecdecode(
  source?: WasmSource,
): Promise<SpecdecodeInstance>;
