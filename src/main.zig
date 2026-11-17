const std = @import("std");
pub const decode_prune = @import("decode_prune.zig");
pub const kv_evict = @import("kv_evict.zig");
pub const cascade_router = @import("cascade_router.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const args = try std.process.argsAlloc(alloc);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "bench")) {
        var timer = try std.time.Timer.start();

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

        timer.reset();
        var i: usize = 0;
        var sink: usize = 0;
        while (i < 10_000) : (i += 1) sink +%= pruner.buildLogitMask(st, &vocab, &mask);
        std.mem.doNotOptimizeAway(sink);
        const prune_ns = timer.read() / 10_000;

        const logits = try alloc.alloc(f32, 128_256);
        @memset(logits, 1.0);
        const words = try alloc.alloc(u64, (128_256 + 63) / 64);
        @memset(words, 0xAAAA_AAAA_AAAA_AAAA);
        timer.reset();
        i = 0;
        while (i < 100) : (i += 1) decode_prune.SemanticPruner.maskLogitsF32(logits, words);
        std.mem.doNotOptimizeAway(logits[0]);
        const mask_128k_us = (timer.read() / 100) / 1000;

        var pool = kv_evict.KvPagePool.init(alloc, 64);
        defer pool.deinit();
        timer.reset();
        i = 0;
        while (i < 2000) : (i += 1) {
            var bbuf: [32]u8 = undefined;
            const tok = std.fmt.bufPrint(&bbuf, "blk_{d}", .{i % 96}) catch "blk";
            _ = try pool.admitBlock(tok, (i % 64) == 0, 0.5, 1.1, @intCast(i));
        }
        const kv_ns = timer.read() / 2000;

        var router = cascade_router.CascadeRouter.init(pruner);
        timer.reset();
        i = 0;
        var c_sink: usize = 0;
        while (i < 10_000) : (i += 1) c_sink +%= @intFromEnum(router.routeStep(st, "0.05", 0.90).tier);
        std.mem.doNotOptimizeAway(c_sink);
        const cascade_ns = timer.read() / 10_000;

        var out: [256]u8 = undefined;
        const msg = try std.fmt.bufPrint(
            &out,
            "{{\"prune_ns_per_step\":{d},\"mask_128k_us\":{d},\"kv_evict_ns_per_admit\":{d},\"cascade_ns_per_route\":{d}}}\n",
            .{ prune_ns, mask_128k_us, kv_ns, cascade_ns },
        );
        try std.fs.File.stdout().writeAll(msg);
        return;
    }

    try std.fs.File.stdout().writeAll("specdecode — Mid-Decode Semantic Pruner, 128k Logit-Mask C ABI, KV-Cache Paging & Cascade Router\n\nUsage:\n  specdecode bench\n");
}

test {
    std.testing.refAllDecls(@This());
}
