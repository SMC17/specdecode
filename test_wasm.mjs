import assert from "node:assert/strict";
import fs from "node:fs";

const wasmBytes = fs.readFileSync(new URL("./zig-out/bin/specdecode.wasm", import.meta.url));
const { instance } = await WebAssembly.instantiate(wasmBytes);
const { memory, wasm_mask_logits_f32 } = instance.exports;

const logits = new Float32Array(memory.buffer, 0, 4);
logits.set([1.0, 2.0, 3.0, 4.0]);
const words = new BigUint64Array(memory.buffer, 64, 1);
words[0] = 0b0101n; // allow indices 0 and 2

wasm_mask_logits_f32(0, 4, 64, 1);
assert.equal(logits[0], 1.0);
assert.equal(logits[1], -Infinity);
assert.equal(logits[2], 3.0);
assert.equal(logits[3], -Infinity);
console.log("node wasm specdecode ok:", Array.from(logits));
