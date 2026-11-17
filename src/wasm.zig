const decode_prune = @import("decode_prune.zig");

export fn wasm_mask_logits_f32(logits_ptr: [*]f32, n_logits: usize, words_ptr: [*]const u64, n_words: usize) void {
    decode_prune.SemanticPruner.maskLogitsF32(logits_ptr[0..n_logits], words_ptr[0..n_words]);
}
