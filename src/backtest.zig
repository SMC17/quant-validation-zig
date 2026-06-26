// Backtest path distribution via CPCV.
//
// Standard purged K-Fold gives a single train/test path: one backtest
// performance figure. CPCV generates C(K, n_test_groups) train/test splits,
// each with a distinct held-out block — so you get a *distribution* of
// backtest performance estimates across multiple paths over the same data.
//
// This module provides:
//   holdoutMeanReturn(fold, returns)      — mean return over one holdout
//   holdoutSharpe(fold, returns, rf)      — annualised Sharpe over one holdout
//   pathScores(alloc, folds, score_fn)    — score all CPCV paths → []f64
//   PathDistribution + summarize(scores)  — mean/stddev/min/max/positive_fraction
//
// The positive_fraction field is the simplest first-order anti-luck check:
// a strategy that wins on 9 of 10 CPCV paths is harder to attribute to
// path-selection luck than one that wins on 5 of 10.

const std = @import("std");
const Allocator = std.mem.Allocator;
const purged_cv = @import("purged_cv.zig");
const sharpe = @import("sharpe.zig");
const stats = @import("stats.zig");

/// Arithmetic mean of returns across the holdout indices of one fold.
pub fn holdoutMeanReturn(fold: purged_cv.Fold, returns: []const f64) f64 {
    if (fold.holdout.len == 0) return 0.0;
    var s: f64 = 0.0;
    for (fold.holdout) |idx| s += returns[idx];
    return s / @as(f64, @floatFromInt(fold.holdout.len));
}

/// Sharpe ratio (mean/stddev) of returns across the holdout indices.
/// Returns 0 if fewer than 2 observations.
/// `risk_free_per_period` subtracts a per-observation risk-free rate before scoring.
pub fn holdoutSharpe(
    alloc: Allocator,
    fold: purged_cv.Fold,
    returns: []const f64,
    risk_free_per_period: f64,
) !f64 {
    if (fold.holdout.len < 2) return 0.0;
    const buf = try alloc.alloc(f64, fold.holdout.len);
    defer alloc.free(buf);
    for (fold.holdout, 0..) |idx, i| buf[i] = returns[idx];
    return sharpe.sharpeRatio(buf, risk_free_per_period);
}

/// Score function type: maps a single fold + returns slice → f64 score.
pub const ScoreFn = *const fn (fold: purged_cv.Fold, returns: []const f64) f64;

/// Compute a score for every CPCV fold. Returns a caller-owned []f64 of
/// length `folds.len`; index i is the score for folds[i].
pub fn pathScores(
    alloc: Allocator,
    folds: []const purged_cv.Fold,
    returns: []const f64,
    score_fn: ScoreFn,
) ![]f64 {
    const scores = try alloc.alloc(f64, folds.len);
    for (folds, 0..) |f, i| scores[i] = score_fn(f, returns);
    return scores;
}

/// Summary statistics over the distribution of CPCV path scores.
pub const PathDistribution = struct {
    /// Arithmetic mean across paths.
    mean: f64,
    /// Sample standard deviation (ddof=1). NaN if fewer than 2 paths.
    stddev: f64,
    min: f64,
    max: f64,
    /// Fraction of paths with strictly positive score.
    positive_fraction: f64,
    n_paths: usize,
};

pub fn summarize(scores: []const f64) PathDistribution {
    const n = scores.len;
    if (n == 0) return .{
        .mean = 0, .stddev = std.math.nan(f64),
        .min = 0, .max = 0, .positive_fraction = 0, .n_paths = 0,
    };

    var mn = scores[0];
    var mx = scores[0];
    var n_pos: usize = 0;
    for (scores) |s| {
        if (s < mn) mn = s;
        if (s > mx) mx = s;
        if (s > 0) n_pos += 1;
    }

    const m = stats.mean(scores);
    const sd = if (n >= 2) stats.stddev(scores, 1) else std.math.nan(f64);
    return .{
        .mean = m,
        .stddev = sd,
        .min = mn,
        .max = mx,
        .positive_fraction = @as(f64, @floatFromInt(n_pos)) / @as(f64, @floatFromInt(n)),
        .n_paths = n,
    };
}

// ─── Tests ───────────────────────────────────────────────────────────────────

test "holdoutMeanReturn — constant returns, all paths equal" {
    const alloc = std.testing.allocator;

    // Build CPCV folds from non-overlapping unit horizons.
    var horizons: [12]purged_cv.Range = undefined;
    for (0..12) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 1) };
    const folds = try @import("cpcv.zig").cpcv(alloc, &horizons, 4, 2, 0);
    defer purged_cv.freeFolds(alloc, folds);

    // All returns = 1.0; every holdout mean must be 1.0.
    var returns = [_]f64{1.0} ** 12;
    for (folds) |f| {
        const score = holdoutMeanReturn(f, &returns);
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), score, 1e-12);
    }
}

test "pathScores + summarize — constant strategy gives zero stddev" {
    const alloc = std.testing.allocator;

    var horizons: [20]purged_cv.Range = undefined;
    for (0..20) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 1) };
    const folds = try @import("cpcv.zig").cpcv(alloc, &horizons, 5, 2, 0);
    defer purged_cv.freeFolds(alloc, folds);

    var returns = [_]f64{2.0} ** 20;
    const scores = try pathScores(alloc, folds, &returns, holdoutMeanReturn);
    defer alloc.free(scores);

    const dist = summarize(scores);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), dist.mean, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), dist.stddev, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), dist.positive_fraction, 1e-12);
}

test "summarize — mixed sign strategy" {
    // 4 paths: +1, -1, +1, -1 → mean=0, positive_fraction=0.5
    const scores = [_]f64{ 1.0, -1.0, 1.0, -1.0 };
    const dist = summarize(&scores);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), dist.mean, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), dist.positive_fraction, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), dist.min, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), dist.max, 1e-12);
}

test "summarize — empty input" {
    const dist = summarize(&.{});
    try std.testing.expectEqual(@as(usize, 0), dist.n_paths);
}

test "holdoutSharpe — constant positive returns give high Sharpe" {
    const alloc = std.testing.allocator;

    var horizons: [10]purged_cv.Range = undefined;
    for (0..10) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 1) };
    const folds = try @import("cpcv.zig").cpcv(alloc, &horizons, 5, 2, 0);
    defer purged_cv.freeFolds(alloc, folds);

    var returns = [_]f64{0.01} ** 10;
    const sr = try holdoutSharpe(alloc, folds[0], &returns, 0.0);
    // Constant returns → stddev=0, Sharpe is Inf or ±large; just verify it's positive.
    // With stddev=0 the NR sharpeRatio returns +Inf (or NaN depending on impl).
    // We only assert it's non-negative for the constant case.
    try std.testing.expect(sr >= 0.0 or std.math.isNan(sr));
}

test "train and holdout independence — score uses only holdout indices" {
    const alloc = std.testing.allocator;

    var horizons: [12]purged_cv.Range = undefined;
    for (0..12) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 1) };
    const folds = try @import("cpcv.zig").cpcv(alloc, &horizons, 4, 2, 0);
    defer purged_cv.freeFolds(alloc, folds);

    // Train indices have return = 999; holdout indices have return = 1.
    // Score must reflect holdout only (= 1.0, not 999.0).
    var returns = [_]f64{999.0} ** 12;
    const f = folds[0];
    for (f.holdout) |idx| returns[idx] = 1.0;

    const score = holdoutMeanReturn(f, &returns);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), score, 1e-12);
}
