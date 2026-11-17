const std = @import("std");

/// Entropy-guided KV-cache paging & prefix-deduplication engine for long-horizon agents.
/// - Deduplicates identical token/prefix KV blocks across concurrent agent trajectories via Blake3.
/// - Pins attention-sink blocks (`sink = true`) so base instructions never evict.
/// - Evicts lowest-utility blocks by combining cumulative attention heavy-hitter score + token entropy.
pub const KvBlock = struct {
    id: u32,
    content_hash: [32]u8,
    ref_count: u32,
    is_sink: bool,
    attn_score: f32,
    entropy: f32,
    last_step: u64,
};

pub const KvPagePool = struct {
    allocator: std.mem.Allocator,
    max_blocks: usize,
    blocks: std.ArrayListUnmanaged(KvBlock) = .{},
    hash_index: std.AutoHashMapUnmanaged([32]u8, u32) = .{},
    next_id: u32 = 1,
    dedup_hits: u64 = 0,
    evictions: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, max_blocks: usize) KvPagePool {
        return .{
            .allocator = allocator,
            .max_blocks = max_blocks,
        };
    }

    pub fn deinit(self: *KvPagePool) void {
        self.blocks.deinit(self.allocator);
        self.hash_index.deinit(self.allocator);
    }

    fn utility(b: KvBlock, current_step: u64) f64 {
        if (b.is_sink) return std.math.inf(f64);
        const age: f64 = @floatFromInt(@max(@as(u64, 1), current_step -| b.last_step));
        // Heavy-hitter attention + information entropy, decayed by recency
        return (@as(f64, b.attn_score) + 0.35 * @as(f64, b.entropy)) / @sqrt(age);
    }

    /// Inserts or deduplicates a KV token block. Evicts lowest-utility non-sink block if full.
    pub fn admitBlock(
        self: *KvPagePool,
        token_bytes: []const u8,
        is_sink: bool,
        attn_score: f32,
        entropy: f32,
        step: u64,
    ) !u32 {
        var h: [32]u8 = undefined;
        std.crypto.hash.Blake3.hash(token_bytes, &h, .{});

        if (self.hash_index.get(h)) |existing_id| {
            for (self.blocks.items) |*b| {
                if (b.id == existing_id) {
                    b.ref_count += 1;
                    b.attn_score = @max(b.attn_score, attn_score);
                    b.last_step = step;
                    self.dedup_hits += 1;
                    return existing_id;
                }
            }
        }

        if (self.blocks.items.len >= self.max_blocks) {
            try self.evictOne(step);
        }

        const id = self.next_id;
        self.next_id += 1;
        try self.blocks.append(self.allocator, .{
            .id = id,
            .content_hash = h,
            .ref_count = 1,
            .is_sink = is_sink,
            .attn_score = attn_score,
            .entropy = entropy,
            .last_step = step,
        });
        try self.hash_index.put(self.allocator, h, id);
        return id;
    }

    fn evictOne(self: *KvPagePool, current_step: u64) !void {
        var victim_idx: ?usize = null;
        var min_util: f64 = std.math.inf(f64);
        for (self.blocks.items, 0..) |b, i| {
            if (b.is_sink) continue;
            const u = utility(b, current_step);
            if (u < min_util) {
                min_util = u;
                victim_idx = i;
            }
        }
        const idx = victim_idx orelse return error.AllBlocksPinned;
        const removed = self.blocks.swapRemove(idx);
        _ = self.hash_index.remove(removed.content_hash);
        self.evictions += 1;
    }

    pub fn containsId(self: *const KvPagePool, id: u32) bool {
        for (self.blocks.items) |b| {
            if (b.id == id) return true;
        }
        return false;
    }
};

test "KvPagePool deduplicates shared prefixes, pins sinks, and evicts lowest entropy/attn block" {
    const alloc = std.testing.allocator;
    var pool = KvPagePool.init(alloc, 3);
    defer pool.deinit();

    const sink_id = try pool.admitBlock("system_prompt_prefix", true, 0.1, 0.1, 1);
    const low_id = try pool.admitBlock("filler_tokens_01", false, 0.05, 0.05, 2);
    const high_id = try pool.admitBlock("critical_error_trace", false, 0.95, 2.4, 3);

    // Duplicate shared prefix across second agent -> deduplicates with zero extra block
    const dup_id = try pool.admitBlock("system_prompt_prefix", true, 0.2, 0.1, 4);
    try std.testing.expectEqual(sink_id, dup_id);
    try std.testing.expectEqual(@as(u64, 1), pool.dedup_hits);

    // Admitting a 4th unique block must evict `low_id` while preserving `sink_id` and `high_id`
    const new_id = try pool.admitBlock("new_reasoning_step", false, 0.6, 1.2, 5);
    try std.testing.expectEqual(@as(u64, 1), pool.evictions);
    try std.testing.expect(pool.containsId(sink_id));
    try std.testing.expect(pool.containsId(high_id));
    try std.testing.expect(pool.containsId(new_id));
    try std.testing.expect(!pool.containsId(low_id));
}
