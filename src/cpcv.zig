const std = @import("std");
const purged_cv = @import("purged_cv.zig");

// Combinatorial Purged Cross-Validation (López de Prado, Advances in
// Financial Machine Learning §7.5).
//
// Standard K-Fold (or Purged K-Fold) gives a single train/test path:
// each observation appears in exactly one test fold. CPCV instead
// enumerates every way to choose `n_test_groups` of the K folds as
// the test set, generating C(K, n_test_groups) splits in total. With
// the same purging + embargo machinery applied per split, the result
// is a Monte-Carlo-style distribution of backtest paths over the same
// underlying data — a defence against the single-path luck that
// Purged K-Fold still admits.
//
// Inputs:
//   horizons:      for each observation i, the half-open interval
//                  [t0_i, t1_i) spanned by its label. Sorted by t0.
//   k:             total number of folds the observations are split into
//   n_test_groups: number of those folds combined into each test set
//                  (1 < n_test_groups < k)
//   embargo:       number of time units to embargo *after* each
//                  contiguous test block (matches purgedKFold semantics)
//
// Output: slice of `purged_cv.Fold` of length C(k, n_test_groups),
// owned by the caller (free via `purged_cv.freeFolds`).

const Range = purged_cv.Range;
const Fold = purged_cv.Fold;

fn binomial(n: usize, k: usize) usize {
    if (k > n) return 0;
    var kk = k;
    if (kk > n - kk) kk = n - kk;
    var num: usize = 1;
    var den: usize = 1;
    var i: usize = 0;
    while (i < kk) : (i += 1) {
        num *= (n - i);
        den *= (i + 1);
    }
    return num / den;
}

pub fn cpcv(
    allocator: std.mem.Allocator,
    horizons: []const Range,
    k: usize,
    n_test_groups: usize,
    embargo: u64,
) ![]Fold {
    std.debug.assert(k >= 2);
    std.debug.assert(n_test_groups >= 1);
    std.debug.assert(n_test_groups < k);
    std.debug.assert(horizons.len >= k);

    if (horizons.len > 1) {
        var prev = horizons[0].t0;
        for (horizons[1..]) |h| {
            std.debug.assert(h.t0 >= prev);
            prev = h.t0;
        }
    }

    const n = horizons.len;

    // Contiguous fold boundaries — identical construction to purgedKFold
    // so a CPCV with n_test_groups == 1 collapses to purged K-Fold.
    var bounds = try allocator.alloc(usize, k + 1);
    defer allocator.free(bounds);
    for (0..k + 1) |i| bounds[i] = (i * n) / k;

    const num_splits = binomial(k, n_test_groups);
    var folds = try allocator.alloc(Fold, num_splits);
    var folds_built: usize = 0;
    errdefer {
        for (folds[0..folds_built]) |f| {
            allocator.free(f.train);
            allocator.free(f.holdout);
        }
        allocator.free(folds);
    }

    // Enumerate combinations of test-group indices via a lexicographic
    // walk over n_test_groups positions in [0, k).
    var combo = try allocator.alloc(usize, n_test_groups);
    defer allocator.free(combo);
    for (0..n_test_groups) |i| combo[i] = i;

    while (true) {
        // Build holdout = union of fold blocks [bounds[g], bounds[g+1])
        // for each g in combo.
        var test_n: usize = 0;
        for (combo) |g| test_n += bounds[g + 1] - bounds[g];
        var holdout_buf = try allocator.alloc(usize, test_n);
        {
            var w: usize = 0;
            for (combo) |g| {
                var i = bounds[g];
                while (i < bounds[g + 1]) : (i += 1) {
                    holdout_buf[w] = i;
                    w += 1;
                }
            }
        }

        // Mark contiguous test blocks. Adjacent fold groups merge into
        // a single block so the embargo gets applied once at the trailing
        // edge of the merged block, not at every internal seam.
        // A block is [block_lo, block_hi) over observation indices.
        var block_los: std.ArrayList(usize) = .empty;
        defer block_los.deinit(allocator);
        var block_his: std.ArrayList(usize) = .empty;
        defer block_his.deinit(allocator);

        var i: usize = 0;
        while (i < combo.len) {
            const start_g = combo[i];
            var end_g = start_g;
            var j = i + 1;
            while (j < combo.len and combo[j] == end_g + 1) : (j += 1) {
                end_g = combo[j];
            }
            try block_los.append(allocator, bounds[start_g]);
            try block_his.append(allocator, bounds[end_g + 1]);
            i = j;
        }

        // Build train: any observation that is (a) not in the holdout
        // AND (b) whose horizon does not overlap any test block's
        // [window_t0, window_t1 + embargo) span.
        var train: std.ArrayList(usize) = .empty;
        defer train.deinit(allocator);
        try train.ensureTotalCapacity(allocator, n - test_n);

        obs_loop: for (0..n) |idx| {
            // Skip if in any test block.
            for (block_los.items, block_his.items) |lo, hi| {
                if (idx >= lo and idx < hi) continue :obs_loop;
            }
            const h = horizons[idx];
            // Reject if horizon overlaps any test block + embargo window.
            for (block_los.items, block_his.items) |lo, hi| {
                var win_t0 = horizons[lo].t0;
                var win_t1 = horizons[lo].t1;
                var bi = lo;
                while (bi < hi) : (bi += 1) {
                    if (horizons[bi].t0 < win_t0) win_t0 = horizons[bi].t0;
                    if (horizons[bi].t1 > win_t1) win_t1 = horizons[bi].t1;
                }
                const embargo_t1 = win_t1 + embargo;
                const disjoint = (h.t1 <= win_t0) or (h.t0 >= embargo_t1);
                if (!disjoint) continue :obs_loop;
            }
            try train.append(allocator, idx);
        }

        folds[folds_built] = .{
            .train = try train.toOwnedSlice(allocator),
            .holdout = holdout_buf,
        };
        folds_built += 1;

        // Lexicographic next-combination, terminating when exhausted.
        var p: isize = @as(isize, @intCast(n_test_groups)) - 1;
        while (p >= 0) : (p -= 1) {
            const pu: usize = @intCast(p);
            const limit: usize = k - (n_test_groups - pu);
            if (combo[pu] < limit) {
                combo[pu] += 1;
                var q: usize = pu + 1;
                while (q < n_test_groups) : (q += 1) combo[q] = combo[q - 1] + 1;
                break;
            }
        }
        if (p < 0) break;
    }

    std.debug.assert(folds_built == num_splits);
    return folds;
}

// ---- internal tests ----

test "cpcv — N=10, K=5, n_test_groups=2 emits C(5,2)=10 splits" {
    const allocator = std.testing.allocator;
    var horizons: [10]Range = undefined;
    for (0..10) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 1) };

    const folds = try cpcv(allocator, &horizons, 5, 2, 0);
    defer purged_cv.freeFolds(allocator, folds);

    try std.testing.expectEqual(@as(usize, 10), folds.len);
}

test "cpcv — each split holds out n_test_groups * (N/K) observations" {
    const allocator = std.testing.allocator;
    var horizons: [20]Range = undefined;
    for (0..20) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 1) };

    const folds = try cpcv(allocator, &horizons, 4, 2, 0);
    defer purged_cv.freeFolds(allocator, folds);

    // N=20, K=4 → fold size 5; n_test_groups=2 → 10 per holdout; C(4,2)=6 splits.
    try std.testing.expectEqual(@as(usize, 6), folds.len);
    for (folds) |f| {
        try std.testing.expectEqual(@as(usize, 10), f.holdout.len);
    }
}

test "cpcv — train and holdout never overlap within a split" {
    const allocator = std.testing.allocator;
    var horizons: [15]Range = undefined;
    for (0..15) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 1) };

    const folds = try cpcv(allocator, &horizons, 5, 2, 0);
    defer purged_cv.freeFolds(allocator, folds);

    for (folds) |f| {
        for (f.train) |tr| {
            for (f.holdout) |te| try std.testing.expect(tr != te);
        }
    }
}

test "cpcv — purging removes overlapping label horizons" {
    const allocator = std.testing.allocator;
    // Each observation's label spans 3 time units (overlaps neighbours).
    var horizons: [10]Range = undefined;
    for (0..10) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 3) };

    const folds = try cpcv(allocator, &horizons, 5, 2, 0);
    defer purged_cv.freeFolds(allocator, folds);

    // Find the split whose test groups are {0, 1} → contiguous holdout
    // = observations {0,1,2,3}. Window = [0, 6). Observation 4 has
    // horizon [4,7), which overlaps the window → must be purged.
    // Observation 5 has [5,8) → overlaps → purged. Observation 6 has
    // [6,9) → t0=6 >= window_t1=6 → disjoint → kept.
    //
    // The first split in lex order has combo = {0, 1}.
    try std.testing.expectEqual(@as(usize, 4), folds[0].holdout.len);
    // Expected survivors: {6, 7, 8, 9} → 4 train observations.
    try std.testing.expectEqual(@as(usize, 4), folds[0].train.len);
    for (folds[0].train) |idx| try std.testing.expect(idx >= 6);
}

test "cpcv — n_test_groups=1 reduces to purged K-Fold split count" {
    const allocator = std.testing.allocator;
    var horizons: [12]Range = undefined;
    for (0..12) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 1) };

    const folds = try cpcv(allocator, &horizons, 4, 1, 0);
    defer purged_cv.freeFolds(allocator, folds);

    // C(4,1) = 4 splits, each with N/K = 3 observations in test.
    try std.testing.expectEqual(@as(usize, 4), folds.len);
    for (folds) |f| {
        try std.testing.expectEqual(@as(usize, 3), f.holdout.len);
    }
}

test "cpcv — embargo applied to non-adjacent test blocks individually" {
    const allocator = std.testing.allocator;
    var horizons: [10]Range = undefined;
    for (0..10) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 1) };

    // K=5, n_test_groups=2, embargo=1. The split with combo {0, 2}
    // gives non-adjacent blocks: test = {0,1, 4,5}. Embargo of 1
    // after block 0 ([0,2)) forbids observation at t=2 from train;
    // embargo after block 1 ([4,6)) forbids observation at t=6.
    // Survivors among non-test {2,3,6,7,8,9}: drop 2 and 6 → {3,7,8,9} = 4.
    const folds = try cpcv(allocator, &horizons, 5, 2, 1);
    defer purged_cv.freeFolds(allocator, folds);

    // Locate split with combo {0, 2}: it is the 2nd split in lex order
    // (combo sequence: {0,1},{0,2},{0,3},{0,4},{1,2}...).
    const f = folds[1];
    try std.testing.expectEqual(@as(usize, 4), f.holdout.len);
    try std.testing.expectEqual(@as(usize, 4), f.train.len);
}
