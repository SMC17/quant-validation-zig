const std = @import("std");

// Purged K-Fold cross-validation generator (López de Prado, Advances in
// Financial Machine Learning §7.4).
//
// Standard K-Fold leaks information when observation labels overlap in
// time. Purged K-Fold removes any training observation whose label
// horizon overlaps the test fold, plus an additional embargo window
// after the test fold to handle serial correlation.
//
// Inputs:
//   horizons:  for each observation i, the half-open interval [t0_i, t1_i)
//              spanned by its label. Must be sorted by t0.
//   k:         number of folds
//   embargo:   number of time units to embargo *after* each test fold
//
// Output: a slice of `Fold` with train_indices + test_indices, owned by
// the caller (free via `freeFolds`).

pub const Range = struct { t0: u64, t1: u64 };

pub const Fold = struct {
    train: []usize,
    holdout: []usize,
};

pub fn freeFolds(allocator: std.mem.Allocator, folds: []Fold) void {
    for (folds) |f| {
        allocator.free(f.train);
        allocator.free(f.holdout);
    }
    allocator.free(folds);
}

pub fn purgedKFold(
    allocator: std.mem.Allocator,
    horizons: []const Range,
    k: usize,
    embargo: u64,
) ![]Fold {
    std.debug.assert(k >= 2);
    std.debug.assert(horizons.len >= k);

    // Verify horizons are sorted by t0 (precondition).
    if (horizons.len > 1) {
        var prev = horizons[0].t0;
        for (horizons[1..]) |h| {
            std.debug.assert(h.t0 >= prev);
            prev = h.t0;
        }
    }

    const n = horizons.len;
    var folds = try allocator.alloc(Fold, k);
    errdefer allocator.free(folds);

    // Contiguous-by-index fold split (standard AFML construction).
    var fold_idx: usize = 0;
    while (fold_idx < k) : (fold_idx += 1) {
        const test_lo = (fold_idx * n) / k;
        const test_hi = ((fold_idx + 1) * n) / k; // exclusive

        // Test indices = [test_lo, test_hi).
        const test_n = test_hi - test_lo;
        var test_buf = try allocator.alloc(usize, test_n);
        for (0..test_n) |i| test_buf[i] = test_lo + i;

        // Determine the time window spanned by the test fold + embargo.
        var window_t0 = horizons[test_lo].t0;
        var window_t1 = horizons[test_lo].t1;
        for (test_lo..test_hi) |i| {
            if (horizons[i].t0 < window_t0) window_t0 = horizons[i].t0;
            if (horizons[i].t1 > window_t1) window_t1 = horizons[i].t1;
        }
        const embargo_t1 = window_t1 + embargo;

        // Train indices = all observations OUTSIDE [test_lo, test_hi)
        // whose horizon does NOT overlap [window_t0, embargo_t1).
        var train: std.ArrayList(usize) = .empty;
        defer train.deinit(allocator);
        try train.ensureTotalCapacity(allocator, n - test_n);
        for (0..n) |i| {
            if (i >= test_lo and i < test_hi) continue;
            const h = horizons[i];
            // Overlap test: NOT (h.t1 <= window_t0 OR h.t0 >= embargo_t1)
            const disjoint = (h.t1 <= window_t0) or (h.t0 >= embargo_t1);
            if (disjoint) try train.append(allocator, i);
        }

        folds[fold_idx] = .{
            .train = try train.toOwnedSlice(allocator),
            .holdout = test_buf,
        };
    }

    return folds;
}

// ---- internal tests ----

test "purgedKFold — non-overlapping unit horizons reduce to standard K-Fold" {
    const allocator = std.testing.allocator;
    var horizons: [10]Range = undefined;
    for (0..10) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 1) };

    const folds = try purgedKFold(allocator, &horizons, 5, 0);
    defer freeFolds(allocator, folds);

    try std.testing.expectEqual(@as(usize, 5), folds.len);
    for (folds) |f| {
        try std.testing.expectEqual(@as(usize, 2), f.holdout.len);
        // No overlap → all non-test observations stay in train.
        try std.testing.expectEqual(@as(usize, 8), f.train.len);
    }
}

test "purgedKFold — overlapping horizons get purged" {
    const allocator = std.testing.allocator;
    // 10 observations, each with horizon = [i, i+3). Window of 3 means
    // observation i shares time with i-2, i-1, i, i+1, i+2.
    var horizons: [10]Range = undefined;
    for (0..10) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 3) };

    const folds = try purgedKFold(allocator, &horizons, 5, 0);
    defer freeFolds(allocator, folds);

    // Fold 0: test = {0,1}, window = [0, 4). Training observations whose
    //   horizon overlaps [0,4): i=2 ([2,5)) overlaps, i=3 ([3,6)) overlaps.
    //   i=4 ([4,7)) is disjoint (t0=4 >= window_t1=4). So train should
    //   exclude {0, 1, 2, 3} → 6 surviving train indices.
    try std.testing.expectEqual(@as(usize, 6), folds[0].train.len);

    // Fold 4: test = {8,9}, window = [8, 12). i=7 ([7,10)) overlaps,
    //   i=6 ([6,9)) overlaps. So train should exclude {6,7,8,9} → 6.
    try std.testing.expectEqual(@as(usize, 6), folds[4].train.len);
}

test "purgedKFold — embargo removes additional post-fold observations" {
    const allocator = std.testing.allocator;
    var horizons: [10]Range = undefined;
    for (0..10) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 1) };

    // Embargo = 2 → after each test fold, the next 2 time units are also
    // forbidden in train.
    const folds = try purgedKFold(allocator, &horizons, 5, 2);
    defer freeFolds(allocator, folds);

    // Fold 0: test = {0,1}, window = [0,2), embargo_t1 = 4.
    //   Observations 2,3 have t0 in [2,4) → purged. Train size = 6.
    try std.testing.expectEqual(@as(usize, 6), folds[0].train.len);

    // Fold 4: test = {8,9}, window = [8,10), embargo_t1 = 12.
    //   No observations after index 9 to purge. Train size = 8.
    try std.testing.expectEqual(@as(usize, 8), folds[4].train.len);
}
