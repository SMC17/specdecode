const std = @import("std");
pub const decode_prune = @import("decode_prune.zig");

pub export fn specdecode_mask_logits_f32(
    logits_ptr: ?[*]f32,
    n_logits: usize,
    mask_words_ptr: ?[*]const u64,
    n_words: usize,
) callconv(.c) i32 {
    const lp = logits_ptr orelse return -1;
    const mp = mask_words_ptr orelse return -1;
    decode_prune.SemanticPruner.maskLogitsF32(lp[0..n_logits], mp[0..n_words]);
    return 0;
}

test "specdecode_mask_logits_f32 C ABI" {
    var lg = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const mw = [_]u64{0b1010};
    try std.testing.expectEqual(@as(i32, 0), specdecode_mask_logits_f32(&lg, 4, &mw, 1));
    try std.testing.expect(std.math.isInf(lg[0]) and lg[0] < 0);
    try std.testing.expectEqual(@as(f32, 2.0), lg[1]);
    try std.testing.expect(std.math.isInf(lg[2]) and lg[2] < 0);
    try std.testing.expectEqual(@as(f32, 4.0), lg[3]);
}
