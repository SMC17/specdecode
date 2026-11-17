const std = @import("std");
const decode_prune = @import("decode_prune.zig");

/// Self-verifying speculative cascade router.
/// Routes decode steps between a fast local draft model and a frontier verifier:
/// - Tracks draft token semantic validity via `SemanticPruner` and confidence margin.
/// - Maintains an anytime-valid e-process over draft error rate; auto-escalates to the
///   frontier model whenever draft rejections cross statistical significance.
pub const ModelTier = enum(u8) {
    local_draft,
    frontier_verifier,
};

pub const RouteDecision = struct {
    tier: ModelTier,
    escalated_by_verifier: bool,
    escalated_by_e_process: bool,
    e_value: f64,
};

pub const CascadeRouter = struct {
    pruner: decode_prune.SemanticPruner,
    target_error_rate: f64 = 0.05,
    alt_error_rate: f64 = 0.25,
    e_threshold: f64 = 20.0, // alpha <= 0.05
    log_e: f64 = 0.0,
    total_steps: u64 = 0,
    escalations: u64 = 0,

    pub fn init(pruner: decode_prune.SemanticPruner) CascadeRouter {
        return .{ .pruner = pruner };
    }

    pub fn routeStep(
        self: *CascadeRouter,
        state: decode_prune.VerifierState,
        proposed_token: []const u8,
        draft_confidence: f32,
    ) RouteDecision {
        self.total_steps += 1;
        const next_state = self.pruner.stepToken(state, proposed_token);
        const semantic_fail = next_state.invalid or draft_confidence < 0.35;

        // Bernoulli SPRT log-likelihood ratio update
        const p0 = self.target_error_rate;
        const p1 = self.alt_error_rate;
        const lr = if (semantic_fail)
            @log(p1 / p0)
        else
            @log((1.0 - p1) / (1.0 - p0));

        self.log_e = @max(0.0, self.log_e + lr);
        const e_val = @exp(@min(self.log_e, 30.0));
        const e_tripped = e_val >= self.e_threshold;

        if (semantic_fail or e_tripped) {
            self.escalations += 1;
            return .{
                .tier = .frontier_verifier,
                .escalated_by_verifier = semantic_fail,
                .escalated_by_e_process = e_tripped,
                .e_value = e_val,
            };
        }

        return .{
            .tier = .local_draft,
            .escalated_by_verifier = false,
            .escalated_by_e_process = false,
            .e_value = e_val,
        };
    }
};

test "CascadeRouter routes valid high-confidence tokens locally and escalates semantic/drift failures" {
    const syms = [_]decode_prune.Symbol{
        .{ .name = "eval_gate", .ty = .fn_t, .arg_types = &.{ .float_t, .float_t }, .min_num = 0.0, .max_num = 1.0 },
    };
    var router = CascadeRouter.init(decode_prune.SemanticPruner.init(&syms));

    var st = decode_prune.VerifierState{};
    // Clean token -> local_draft
    const d0 = router.routeStep(st, "eval_gate", 0.92);
    try std.testing.expectEqual(ModelTier.local_draft, d0.tier);
    st = router.pruner.stepToken(st, "eval_gate");
    st = router.pruner.stepToken(st, "(");

    // Semantic violation (2.5 > max_num 1.0) -> immediate frontier_verifier escalation
    const d1 = router.routeStep(st, "2.5", 0.88);
    try std.testing.expectEqual(ModelTier.frontier_verifier, d1.tier);
    try std.testing.expect(d1.escalated_by_verifier);

    // Repeated low-confidence tokens trip anytime-valid e-process even if syntax passes
    var tripped_e = false;
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const d = router.routeStep(st, "0.5", 0.20);
        if (d.escalated_by_e_process) {
            tripped_e = true;
            break;
        }
    }
    try std.testing.expect(tripped_e);
}
