// External integration tests — consume the public `quant_validation`
// module the way a downstream caller would. Reference numbers come
// from Bailey & López de Prado 2012/2014 papers and standard normal
// tables; we do not depend on any private implementation detail.

const std = @import("std");
const qv = @import("quant_validation");

test "normal CDF — Phi(1.96) ≈ 0.975 (canonical 95 percent quantile)" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.975), qv.stats.normCdf(1.959963984540054), 1e-7);
}

test "PSR — high-Sharpe normal returns approach 1" {
    // SR_hat = 1.0, n = 37, SR* = 0, normal moments.
    // denom = sqrt(1 + 0.5) = 1.224744871...
    // numer = 1 * sqrt(36) = 6
    // z = 4.898979485..., Phi(z) ≈ 0.99999952
    const p = qv.sharpe.psr(1.0, 0.0, 37, 0.0, 3.0);
    try std.testing.expect(p > 0.9999);
    try std.testing.expect(p < 1.0);
}

test "PSR — fat-tail penalty against same observed SR" {
    // Same SR_hat = 1.0, n = 50 against the benchmark, but skewness
    // and kurtosis worsen. PSR must drop monotonically.
    const a = qv.sharpe.psr(1.0, 0.0, 50, 0.0, 3.0);
    const b = qv.sharpe.psr(1.0, 0.0, 50, -0.2, 3.5);
    const c = qv.sharpe.psr(1.0, 0.0, 50, -0.5, 5.0);
    try std.testing.expect(a > b);
    try std.testing.expect(b > c);
}

test "DSR — increasing num_trials raises the null bar" {
    // V[SR] held constant, SR_hat held constant. As num_trials grows,
    // SR0 grows, so DSR (the prob the observed beats the null max) drops.
    const v: f64 = 0.04;
    const sr_hat: f64 = 0.5;
    const d10 = qv.sharpe.dsr(sr_hat, 250, 0.0, 3.0, v, 10);
    const d100 = qv.sharpe.dsr(sr_hat, 250, 0.0, 3.0, v, 100);
    const d1000 = qv.sharpe.dsr(sr_hat, 250, 0.0, 3.0, v, 1000);
    try std.testing.expect(d10 > d100);
    try std.testing.expect(d100 > d1000);
}

test "expectedMaxSharpe — N=10, V=0.1 reproduces published reference" {
    // Bailey & López de Prado 2014, worked example used in the AFML
    // companion code. SR0 ≈ 0.4976 (within 1e-3 of the paper's value).
    const sr0 = qv.sharpe.expectedMaxSharpe(0.1, 10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4976), sr0, 1e-3);
}

test "purgedKFold — total test indices cover all observations" {
    const allocator = std.testing.allocator;
    var horizons: [20]qv.purged_cv.Range = undefined;
    for (0..20) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 1) };

    const folds = try qv.purged_cv.purgedKFold(allocator, &horizons, 4, 0);
    defer qv.purged_cv.freeFolds(allocator, folds);

    var seen = [_]bool{false} ** 20;
    for (folds) |f| {
        for (f.holdout) |idx| {
            try std.testing.expect(!seen[idx]);
            seen[idx] = true;
        }
    }
    for (seen) |s| try std.testing.expect(s);
}

test "purgedKFold — train and holdout never overlap" {
    const allocator = std.testing.allocator;
    var horizons: [15]qv.purged_cv.Range = undefined;
    for (0..15) |i| horizons[i] = .{ .t0 = @intCast(i * 2), .t1 = @intCast(i * 2 + 5) };

    const folds = try qv.purged_cv.purgedKFold(allocator, &horizons, 3, 1);
    defer qv.purged_cv.freeFolds(allocator, folds);

    for (folds) |f| {
        for (f.train) |tr| {
            for (f.holdout) |te| try std.testing.expect(tr != te);
        }
    }
}

test "sharpeRatio — constant excess returns give large positive SR" {
    var rets: [50]f64 = undefined;
    for (&rets, 0..) |*r, i| r.* = if (i % 2 == 0) 0.011 else 0.009;
    const sr = qv.sharpe.sharpeRatio(&rets, 0.0);
    try std.testing.expect(sr > 5.0);
}

test "cpcv — split count matches C(K, n_test_groups) via public API" {
    const allocator = std.testing.allocator;
    var horizons: [24]qv.purged_cv.Range = undefined;
    for (0..24) |i| horizons[i] = .{ .t0 = @intCast(i), .t1 = @intCast(i + 1) };

    // C(6, 2) = 15
    const folds = try qv.cpcv.cpcv(allocator, &horizons, 6, 2, 0);
    defer qv.purged_cv.freeFolds(allocator, folds);
    try std.testing.expectEqual(@as(usize, 15), folds.len);
}
