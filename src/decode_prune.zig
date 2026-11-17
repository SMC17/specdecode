const std = @import("std");

/// Mid-decode incremental semantic & type-state pruner for test-time compute.
/// Runs inside speculative/token-tree decoding loops in microseconds per step:
/// - Tracks declared symbols and typed function signatures.
/// - Enforces call arity, argument types, balanced delimiters, and numeric contracts.
/// - Emits packed u64 bitsets over candidate vocab tokens to prune ill-typed branches mid-decode.
pub const ValType = enum(u8) {
    any,
    int_t,
    float_t,
    str_t,
    bool_t,
    fn_t,
};

pub const Symbol = struct {
    name: []const u8,
    ty: ValType,
    arg_types: []const ValType = &.{},
    ret_type: ValType = .any,
    min_num: ?f64 = null,
    max_num: ?f64 = null,
};

pub const DecodePhase = enum(u8) {
    stmt_start,
    expect_lparen,
    expect_arg,
    expect_comma_or_rparen,
    closed,
};

pub const VerifierState = struct {
    phase: DecodePhase = .stmt_start,
    active_fn_idx: ?u16 = null,
    arg_pos: u8 = 0,
    paren_depth: u8 = 0,
    brace_depth: u8 = 0,
    invalid: bool = false,
};

pub const SemanticPruner = struct {
    symbols: []const Symbol,

    pub fn init(symbols: []const Symbol) SemanticPruner {
        return .{ .symbols = symbols };
    }

    pub fn findSymbol(self: *const SemanticPruner, name: []const u8) ?u16 {
        for (self.symbols, 0..) |s, i| {
            if (std.mem.eql(u8, s.name, name)) return @intCast(i);
        }
        return null;
    }

    /// Evaluates whether advancing `state` by `token` preserves semantic/type validity.
    /// Zero heap allocations; safe for O(1) speculative beam/tree rollbacks.
    pub fn stepToken(self: *const SemanticPruner, state: VerifierState, token: []const u8) VerifierState {
        if (state.invalid) return state;
        const tok = std.mem.trim(u8, token, " \t\r\n");
        if (tok.len == 0) return state;

        var next = state;
        switch (state.phase) {
            .stmt_start => {
                if (self.findSymbol(tok)) |idx| {
                    const sym = self.symbols[idx];
                    if (sym.ty == .fn_t) {
                        next.active_fn_idx = idx;
                        next.arg_pos = 0;
                        next.phase = .expect_lparen;
                        return next;
                    }
                }
                next.invalid = true;
                return next;
            },
            .expect_lparen => {
                if (std.mem.eql(u8, tok, "(")) {
                    next.paren_depth += 1;
                    const fn_sym = self.symbols[state.active_fn_idx.?];
                    next.phase = if (fn_sym.arg_types.len == 0) .expect_comma_or_rparen else .expect_arg;
                    return next;
                }
                next.invalid = true;
                return next;
            },
            .expect_arg => {
                const fn_sym = self.symbols[state.active_fn_idx.?];
                if (state.arg_pos >= fn_sym.arg_types.len) {
                    next.invalid = true;
                    return next;
                }
                const expected = fn_sym.arg_types[state.arg_pos];
                if (!self.tokenMatchesType(tok, expected, fn_sym.min_num, fn_sym.max_num)) {
                    next.invalid = true;
                    return next;
                }
                next.arg_pos += 1;
                next.phase = .expect_comma_or_rparen;
                return next;
            },
            .expect_comma_or_rparen => {
                const fn_sym = self.symbols[state.active_fn_idx.?];
                if (std.mem.eql(u8, tok, ",")) {
                    if (state.arg_pos >= fn_sym.arg_types.len) {
                        next.invalid = true;
                        return next;
                    }
                    next.phase = .expect_arg;
                    return next;
                }
                if (std.mem.eql(u8, tok, ")")) {
                    if (state.paren_depth == 0 or state.arg_pos != fn_sym.arg_types.len) {
                        next.invalid = true;
                        return next;
                    }
                    next.paren_depth -= 1;
                    next.phase = .closed;
                    return next;
                }
                next.invalid = true;
                return next;
            },
            .closed => {
                if (std.mem.eql(u8, tok, ";")) return .{};
                next.invalid = true;
                return next;
            },
        }
    }

    fn tokenMatchesType(self: *const SemanticPruner, tok: []const u8, expected: ValType, min_num: ?f64, max_num: ?f64) bool {
        if (expected == .any) return true;
        // Check typed variable reference in symbol table
        if (self.findSymbol(tok)) |idx| {
            const sym = self.symbols[idx];
            if (sym.ty != .fn_t and (sym.ty == expected or (expected == .float_t and sym.ty == .int_t))) {
                return true;
            }
            return false;
        }
        return switch (expected) {
            .bool_t => std.mem.eql(u8, tok, "true") or std.mem.eql(u8, tok, "false"),
            .str_t => tok.len >= 2 and tok[0] == '"' and tok[tok.len - 1] == '"',
            .int_t => blk: {
                const v = std.fmt.parseInt(i64, tok, 10) catch break :blk false;
                const vf: f64 = @floatFromInt(v);
                if (min_num) |lo| if (vf < lo) break :blk false;
                if (max_num) |hi| if (vf > hi) break :blk false;
                break :blk true;
            },
            .float_t => blk: {
                const vf = std.fmt.parseFloat(f64, tok) catch break :blk false;
                if (std.math.isNan(vf) or std.math.isInf(vf)) break :blk false;
                if (min_num) |lo| if (vf < lo) break :blk false;
                if (max_num) |hi| if (vf > hi) break :blk false;
                break :blk true;
            },
            .any => true,
            .fn_t => false,
        };
    }

    /// Populates packed 64-bit logit mask words (`1` = allowed, `0` = pruned).
    /// Returns number of surviving valid candidate tokens.
    pub fn buildLogitMask(
        self: *const SemanticPruner,
        state: VerifierState,
        vocab: []const []const u8,
        mask_words: []u64,
    ) usize {
        @memset(mask_words, 0);
        var allowed: usize = 0;
        for (vocab, 0..) |tok, i| {
            const res = self.stepToken(state, tok);
            if (!res.invalid) {
                mask_words[i >> 6] |= (@as(u64, 1) << @as(u6, @intCast(i & 63)));
                allowed += 1;
            }
        }
        return allowed;
    }

    /// Applies packed 64-bit logit mask directly to a 128k+ f32 logit vector in-place
    /// using portable 4-lane SIMD (@Vector(4, f32)) selection. Disallowed tokens become -inf.
    pub fn maskLogitsF32(logits: []f32, mask_words: []const u64) void {
        const neg_inf_val = -std.math.inf(f32);
        const neg_inf_vec: @Vector(4, f32) = @splat(neg_inf_val);
        const bit_masks: @Vector(4, u32) = .{ 1, 2, 4, 8 };
        const zero_u32: @Vector(4, u32) = @splat(0);
        const n_words = @min(mask_words.len, (logits.len + 63) >> 6);

        var w: usize = 0;
        while (w < n_words) : (w += 1) {
            const word = mask_words[w];
            const base = w << 6;
            const end = @min(base + 64, logits.len);
            if (word == std.math.maxInt(u64)) continue;
            if (word == 0) {
                @memset(logits[base..end], neg_inf_val);
                continue;
            }
            var off: usize = 0;
            while (base + off + 4 <= end) : (off += 4) {
                const nib: u32 = @intCast((word >> @as(u6, @intCast(off))) & 0xF);
                const splat_nib: @Vector(4, u32) = @splat(nib);
                const keep = (splat_nib & bit_masks) != zero_u32;
                const v: @Vector(4, f32) = logits[base + off ..][0..4].*;
                logits[base + off ..][0..4].* = @select(f32, keep, v, neg_inf_vec);
            }
            while (base + off < end) : (off += 1) {
                if ((word & (@as(u64, 1) << @as(u6, @intCast(off)))) == 0) {
                    logits[base + off] = neg_inf_val;
                }
            }
        }
    }
};

test "SemanticPruner masks invalid symbols, arity, types, and range contracts mid-decode" {
    const syms = [_]Symbol{
        .{
            .name = "eval_gate",
            .ty = .fn_t,
            .arg_types = &.{ .float_t, .float_t },
            .ret_type = .bool_t,
            .min_num = 0.0,
            .max_num = 1.0,
        },
        .{ .name = "delta", .ty = .float_t },
        .{ .name = "bad_str", .ty = .str_t },
    };
    const pruner = SemanticPruner.init(&syms);

    const vocab = [_][]const u8{
        "eval_gate",
        "unknown_fn",
        "(",
        ")",
        ",",
        "0.05",
        "1.50",
        "\"oops\"",
        "delta",
        "bad_str",
    };

    var mask: [1]u64 = undefined;
    var st = VerifierState{};

    // Step 0: only `eval_gate` is a valid callable symbol
    try std.testing.expectEqual(@as(usize, 1), pruner.buildLogitMask(st, &vocab, &mask));
    try std.testing.expect((mask[0] & 1) != 0);

    st = pruner.stepToken(st, "eval_gate");
    st = pruner.stepToken(st, "(");

    // Expect arg 0 (float in [0, 1] or float symbol `delta`)
    const n_arg0 = pruner.buildLogitMask(st, &vocab, &mask);
    try std.testing.expectEqual(@as(usize, 2), n_arg0); // "0.05" and "delta" (1.50 violates max_num=1.0, bad_str wrong type)

    st = pruner.stepToken(st, "0.05");
    // Premature ")" must be masked out because arity requires 2 args
    try std.testing.expectEqual(@as(usize, 1), pruner.buildLogitMask(st, &vocab, &mask)); // only ","

    st = pruner.stepToken(st, ",");
    st = pruner.stepToken(st, "delta");
    // Now only ")" is valid
    try std.testing.expectEqual(@as(usize, 1), pruner.buildLogitMask(st, &vocab, &mask));
    st = pruner.stepToken(st, ")");
    try std.testing.expectEqual(DecodePhase.closed, st.phase);
    try std.testing.expect(!st.invalid);

    // Verify 128,256-vocab f32 logit masking
    const alloc = std.testing.allocator;
    const logits = try alloc.alloc(f32, 128_256);
    defer alloc.free(logits);
    @memset(logits, 1.0);
    const words = try alloc.alloc(u64, (128_256 + 63) / 64);
    defer alloc.free(words);
    @memset(words, 0);
    words[0] = 0b101; // only token 0 and 2 allowed
    SemanticPruner.maskLogitsF32(logits, words);
    try std.testing.expectEqual(@as(f32, 1.0), logits[0]);
    try std.testing.expect(std.math.isInf(logits[1]) and logits[1] < 0);
    try std.testing.expectEqual(@as(f32, 1.0), logits[2]);
    try std.testing.expect(std.math.isInf(logits[128_000]) and logits[128_000] < 0);
}
