const WASM_PAGE_SIZE = 65536;

function isNodeRuntime() {
  return (
    typeof process !== "undefined" &&
    process.versions != null &&
    process.versions.node != null
  );
}

async function resolveDefaultWasmBytes() {
  const localUrl = new URL("./specdecode.wasm", import.meta.url);
  if (isNodeRuntime()) {
    const fs = await import("node:fs/promises");
    try {
      return await fs.readFile(localUrl);
    } catch {
      const repoBuildUrl = new URL("../zig-out/bin/specdecode.wasm", import.meta.url);
      return await fs.readFile(repoBuildUrl);
    }
  }
  return await fetch(localUrl);
}

async function instantiateWasm(source) {
  const resolved = source ?? (await resolveDefaultWasmBytes());

  if (resolved instanceof WebAssembly.Module) {
    return await WebAssembly.instantiate(resolved);
  }

  if (typeof Response !== "undefined" && resolved instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming === "function") {
      try {
        const { instance } = await WebAssembly.instantiateStreaming(resolved.clone());
        return instance;
      } catch {
        // Fallback when MIME type is not application/wasm
      }
    }
    const bytes = await resolved.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes);
    return instance;
  }

  if (typeof resolved === "string" || resolved instanceof URL) {
    if (isNodeRuntime()) {
      const fs = await import("node:fs/promises");
      const bytes = await fs.readFile(resolved);
      const { instance } = await WebAssembly.instantiate(bytes);
      return instance;
    }
    const resp = await fetch(resolved);
    return await instantiateWasm(resp);
  }

  const { instance } = await WebAssembly.instantiate(resolved);
  return instance;
}

export async function createSpecdecode(source) {
  const instance = await instantiateWasm(source);
  const exports = instance.exports;
  const memory = exports.memory;
  const wasmMask = exports.wasm_mask_logits_f32;

  function ensureMemory(minBytes) {
    const current = memory.buffer.byteLength;
    if (minBytes <= current) return;
    const pagesNeeded = Math.ceil((minBytes - current) / WASM_PAGE_SIZE);
    memory.grow(pagesNeeded);
  }

  function maskLogitsInPlace(logitsPtr, nLogits, wordsPtr, nWords) {
    wasmMask(logitsPtr, nLogits, wordsPtr, nWords);
  }

  function maskLogits(logits, maskWords) {
    const wordsOffset = (logits.byteLength + 7) & ~7;
    const totalBytes = wordsOffset + maskWords.byteLength;
    ensureMemory(totalBytes);

    const wasmLogits = new Float32Array(memory.buffer, 0, logits.length);
    wasmLogits.set(logits);

    const wasmWords = new BigUint64Array(memory.buffer, wordsOffset, maskWords.length);
    wasmWords.set(maskWords);

    wasmMask(0, logits.length, wordsOffset, maskWords.length);

    logits.set(new Float32Array(memory.buffer, 0, logits.length));
    return logits;
  }

  return {
    memory,
    exports,
    ensureMemory,
    maskLogits,
    maskLogitsInPlace,
  };
}
