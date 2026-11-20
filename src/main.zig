const std = @import("std");
pub const decode_prune = @import("decode_prune.zig");
pub const kv_evict = @import("kv_evict.zig");
pub const cascade_router = @import("cascade_router.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const args = try init.minimal.args.toSlice(alloc);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "bench")) {
        var start: std.Io.Timestamp = undefined;

        const syms = [_]decode_prune.Symbol{
            .{ .name = "eval_gate", .ty = .fn_t, .arg_types = &.{ .float_t, .float_t }, .min_num = 0.0, .max_num = 1.0 },
            .{ .name = "delta", .ty = .float_t },
        };
        const pruner = decode_prune.SemanticPruner.init(&syms);
        const vocab = [_][]const u8{ "eval_gate", "(", ")", ",", "0.05", "1.50", "delta", "\"bad\"" };
        var mask: [1]u64 = undefined;
        var st = decode_prune.VerifierState{};
        st = pruner.stepToken(st, "eval_gate");
        st = pruner.stepToken(st, "(");

        start = std.Io.Timestamp.now(io, .awake);
        var i: usize = 0;
        var sink: usize = 0;
        while (i < 10_000) : (i += 1) sink +%= pruner.buildLogitMask(st, &vocab, &mask);
        std.mem.doNotOptimizeAway(sink);
        const prune_ns = elapsedNs(io, start) / 10_000;

        const logits = try alloc.alloc(f32, 128_256);
        @memset(logits, 1.0);
        const words = try alloc.alloc(u64, (128_256 + 63) / 64);
        @memset(words, 0xAAAA_AAAA_AAAA_AAAA);
        start = std.Io.Timestamp.now(io, .awake);
        i = 0;
        while (i < 100) : (i += 1) decode_prune.SemanticPruner.maskLogitsF32(logits, words);
        std.mem.doNotOptimizeAway(logits[0]);
        const mask_128k_us = (elapsedNs(io, start) / 100) / 1000;

        var pool = kv_evict.KvPagePool.init(alloc, 64);
        defer pool.deinit();
        start = std.Io.Timestamp.now(io, .awake);
        i = 0;
        while (i < 2000) : (i += 1) {
            var bbuf: [32]u8 = undefined;
            const tok = std.fmt.bufPrint(&bbuf, "blk_{d}", .{i % 96}) catch "blk";
            _ = try pool.admitBlock(tok, (i % 64) == 0, 0.5, 1.1, @intCast(i));
        }
        const kv_ns = elapsedNs(io, start) / 2000;

        // Pruner on a vocabulary of realistic size: 32,000 whole-word tokens
        // (identifiers, numbers, punctuation), one mask per decode step.
        const big_n = 32_000;
        const big_vocab = try alloc.alloc([]const u8, big_n);
        for (big_vocab, 0..) |*t, k| {
            t.* = switch (k % 4) {
                0 => try std.fmt.allocPrint(alloc, "ident{d}", .{k}),
                1 => try std.fmt.allocPrint(alloc, "0.{d}", .{k}),
                2 => if (k % 8 == 2) "," else ")",
                else => if (k == 3) "delta" else try std.fmt.allocPrint(alloc, "\"s{d}\"", .{k}),
            };
        }
        const big_mask = try alloc.alloc(u64, (big_n + 63) / 64);
        start = std.Io.Timestamp.now(io, .awake);
        i = 0;
        while (i < 200) : (i += 1) sink +%= pruner.buildLogitMask(st, big_vocab, big_mask);
        std.mem.doNotOptimizeAway(sink);
        const prune_32k_us = (elapsedNs(io, start) / 200) / 1000;

        var router = cascade_router.CascadeRouter.init(pruner);
        start = std.Io.Timestamp.now(io, .awake);
        i = 0;
        var c_sink: usize = 0;
        while (i < 10_000) : (i += 1) c_sink +%= @intFromEnum(router.routeStep(st, "0.05", 0.90).tier);
        std.mem.doNotOptimizeAway(c_sink);
        const cascade_ns = elapsedNs(io, start) / 10_000;

        var out: [256]u8 = undefined;
        const msg = try std.fmt.bufPrint(
            &out,
            "{{\"prune_8tok_ns_per_step\":{d},\"prune_32k_vocab_us_per_step\":{d},\"mask_128k_us\":{d},\"kv_evict_ns_per_admit\":{d},\"cascade_ns_per_route\":{d}}}\n",
            .{ prune_ns, prune_32k_us, mask_128k_us, kv_ns, cascade_ns },
        );
        try std.Io.File.stdout().writeStreamingAll(io, msg);
        return;
    }

    try std.Io.File.stdout().writeStreamingAll(io, "specdecode — Mid-Decode Semantic Pruner, 128k Logit-Mask C ABI, KV-Cache Paging & Cascade Router\n\nUsage:\n  specdecode bench\n");
}

/// Nanoseconds elapsed on the monotonic clock since `start` (replaces the removed `std.time.Timer.read`).
fn elapsedNs(io: std.Io, start: std.Io.Timestamp) u64 {
    const ns = start.untilNow(io, .awake).nanoseconds;
    return @intCast(@max(ns, 0));
}

test {
    std.testing.refAllDecls(@This());
}
