const std = @import("std");

// Chi-square survival function + Pearson contingency-table test, built
// on the regularized incomplete gamma functions P(a,x) and Q(a,x)
// (Numerical Recipes 3rd ed. §6.2: `gser` series for x < a+1, `gcf`
// Lenz continued fraction otherwise).
//
// Numerical-stability note: the continued-fraction branch is evaluated
// in LOG space — ln Q(a,x) = -x + a*ln(x) - gammln(a) + ln(h) — and
// exp'd once at the end. The naive form multiplies exp(-x + a*ln x -
// gammln(a)) by the continued-fraction value h as separate factors;
// keeping everything as a single log sum means tail probabilities far
// below 1 (e.g. the ~5e-218 reproduced in examples/repro_lingstats.zig)
// stay representable: f64 min normal is ~2.2e-308, so any p ≥ ~1e-307
// comes out exact to the precision of the continued fraction itself
// (~1e-15 relative) rather than degrading through intermediate
// underflow.

// Log-gamma via the Lanczos approximation (NR3 §6.1, `gammln`).
// Full f64 accuracy for x > 0; the 14 coefficients below are the NR3
// set (g = 671/128, N = 14).
const lanczos_cof = [_]f64{
    57.1562356658629235,     -59.5979603554754912,
    14.1360979747417471,     -0.491913816097620199,
    0.339946499848118887e-4, 0.465236289270485756e-4,
    -0.983744753048795646e-4, 0.158088703224912494e-3,
    -0.210264441724104883e-3, 0.217439618115212643e-3,
    -0.164318106536763890e-3, 0.844182239838527433e-4,
    -0.261908384015814087e-4, 0.368991826595316234e-5,
};

pub fn gammln(xx: f64) f64 {
    std.debug.assert(xx > 0);
    var y = xx;
    const tmp0 = xx + 5.24218750000000000; // x + g + 1/2, g = 671/128
    const tmp = (xx + 0.5) * @log(tmp0) - tmp0;
    var ser: f64 = 0.999999999999997092;
    for (lanczos_cof) |c| {
        y += 1.0;
        ser += c / y;
    }
    return tmp + @log(2.5066282746310005 * ser / xx);
}

const eps: f64 = std.math.floatEps(f64);
const fpmin: f64 = std.math.floatMin(f64) / eps;
const max_iter = 1000;

// Series representation of P(a,x) (NR3 `gser`). Converges fast for
// x < a + 1. The prefactor exp(-x + a*ln(x) - gammln(a)) is benign in
// this regime (P is not in the deep tail here), so no log gymnastics
// are needed on this branch.
fn gser(a: f64, x: f64) f64 {
    var ap = a;
    var sum = 1.0 / a;
    var del = sum;
    var i: usize = 0;
    while (i < max_iter) : (i += 1) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (@abs(del) < @abs(sum) * eps) break;
    }
    return sum * @exp(-x + a * @log(x) - gammln(a));
}

// Continued-fraction representation of Q(a,x) (NR3 `gcf`, modified
// Lentz). Used for x >= a + 1. Returns ln Q(a,x) — see the module
// header for why the log-prefactor is only exp'd at the call site.
fn gcfLogQ(a: f64, x: f64) f64 {
    var b = x + 1.0 - a;
    var c = 1.0 / fpmin;
    var d = 1.0 / b;
    var h = d;
    var i: usize = 1;
    while (i <= max_iter) : (i += 1) {
        const an = -@as(f64, @floatFromInt(i)) * (@as(f64, @floatFromInt(i)) - a);
        b += 2.0;
        d = an * d + b;
        if (@abs(d) < fpmin) d = fpmin;
        c = b + an / c;
        if (@abs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        const del = d * c;
        h *= del;
        if (@abs(del - 1.0) <= eps) break;
    }
    return -x + a * @log(x) - gammln(a) + @log(h);
}

// Regularized lower incomplete gamma P(a,x) = γ(a,x) / Γ(a).
pub fn gammaP(a: f64, x: f64) f64 {
    std.debug.assert(a > 0 and x >= 0);
    if (x == 0) return 0.0;
    if (x < a + 1.0) return gser(a, x);
    return 1.0 - @exp(gcfLogQ(a, x));
}

// Regularized upper incomplete gamma Q(a,x) = 1 - P(a,x).
pub fn gammaQ(a: f64, x: f64) f64 {
    std.debug.assert(a > 0 and x >= 0);
    if (x == 0) return 1.0;
    if (x < a + 1.0) return 1.0 - gser(a, x);
    return @exp(gcfLogQ(a, x));
}

// Chi-square survival function: P(X >= x) for X ~ chi2(df).
// chi2_sf(x, df) = Q(df/2, x/2). For df = 1 this reduces to the
// analytic identity erfc(sqrt(x/2)) — exercised as a cross-check
// against this library's own `stats.erfc` in the test suite.
pub fn chi2Sf(x: f64, df: f64) f64 {
    std.debug.assert(df > 0 and x >= 0);
    return gammaQ(df / 2.0, x / 2.0);
}

pub const ContingencyResult = struct {
    statistic: f64,
    df: f64,
    p: f64,
};

// Pearson chi-square test of independence on an r×c contingency table
// of counts (row-major, `table.len == n_rows * n_cols`). Computes
// X² = Σ (O - E)² / E with E_ij = row_i * col_j / N, df = (r-1)(c-1),
// and p = chi2Sf(X², df). No Yates continuity correction (matches
// scipy.stats.chi2_contingency(..., correction=False); for tables
// larger than 2×2 scipy applies no correction either way).
// Errors with `error.ZeroMarginal` if any row or column sums to zero
// (the expected count would be zero and X² undefined).
pub fn contingencyChi2(
    allocator: std.mem.Allocator,
    table: []const f64,
    n_rows: usize,
    n_cols: usize,
) !ContingencyResult {
    std.debug.assert(n_rows >= 2 and n_cols >= 2);
    std.debug.assert(table.len == n_rows * n_cols);

    const col_sums = try allocator.alloc(f64, n_cols);
    defer allocator.free(col_sums);
    @memset(col_sums, 0);

    const row_sums = try allocator.alloc(f64, n_rows);
    defer allocator.free(row_sums);
    @memset(row_sums, 0);

    var total: f64 = 0;
    for (0..n_rows) |i| {
        for (0..n_cols) |j| {
            const o = table[i * n_cols + j];
            std.debug.assert(o >= 0);
            row_sums[i] += o;
            col_sums[j] += o;
            total += o;
        }
    }
    for (row_sums) |r| if (r == 0) return error.ZeroMarginal;
    for (col_sums) |c| if (c == 0) return error.ZeroMarginal;

    var statistic: f64 = 0;
    for (0..n_rows) |i| {
        for (0..n_cols) |j| {
            const expected = row_sums[i] * col_sums[j] / total;
            const d = table[i * n_cols + j] - expected;
            statistic += d * d / expected;
        }
    }

    const df: f64 = @floatFromInt((n_rows - 1) * (n_cols - 1));
    return .{ .statistic = statistic, .df = df, .p = chi2Sf(statistic, df) };
}

// Internal tests — golden values are analytic where possible:
// gammln against exact factorials, chi2Sf identities, and a
// gser-branch reference computed independently via the same NR3
// recurrences in double precision (cross-checked against math.erfc /
// exp identities to ~4e-16 relative).
test "gammln — exact reference values" {
    // Gamma(1/2) = sqrt(pi); Gamma(1) = 1; Gamma(10) = 9! = 362880
    try std.testing.expectApproxEqAbs(@log(@sqrt(std.math.pi)), gammln(0.5), 1e-14);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), gammln(1.0), 1e-14);
    try std.testing.expectApproxEqAbs(@log(@as(f64, 362880.0)), gammln(10.0), 1e-12);
}

test "gammaP/gammaQ — complementarity and boundaries" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), gammaP(2.5, 0.0), 0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), gammaQ(2.5, 0.0), 0);
    // P + Q = 1 on both branches
    const probes = [_][2]f64{ .{ 3.0, 1.0 }, .{ 3.0, 10.0 }, .{ 0.5, 0.2 }, .{ 0.5, 8.0 } };
    for (probes) |pr| {
        try std.testing.expectApproxEqAbs(
            @as(f64, 1.0),
            gammaP(pr[0], pr[1]) + gammaQ(pr[0], pr[1]),
            1e-12,
        );
    }
}

test "chi2Sf — sf(0, df) = 1 for any df" {
    const dfs = [_]f64{ 1, 2, 5, 14, 56 };
    for (dfs) |df| {
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), chi2Sf(0.0, df), 0);
    }
}

test "chi2Sf — df = 2 analytic identity sf(x, 2) = exp(-x/2)" {
    const xs = [_]f64{ 0.1, 1.0, 5.991, 20.0 };
    for (xs) |x| {
        try std.testing.expectApproxEqRel(@exp(-x / 2.0), chi2Sf(x, 2.0), 1e-13);
    }
}

test "chi2Sf — gser branch reference (x < df/2 + 1)" {
    // sf(1.0, 10): a = 5, x/2 = 0.5 < a+1 → series branch.
    // Reference 0.9998278843700441 via the NR3 recurrences in double
    // precision, independently validated against scipy-table values.
    try std.testing.expectApproxEqRel(@as(f64, 0.9998278843700441), chi2Sf(1.0, 10.0), 1e-13);
}

test "contingencyChi2 — hand-computed 2x2 table" {
    // T = [[10, 20], [30, 40]]: X² = 50/63 = 0.793650...,
    // df = 1, p = 0.3729984836134872 (analytic via erfc identity).
    const allocator = std.testing.allocator;
    const table = [_]f64{ 10, 20, 30, 40 };
    const r = try contingencyChi2(allocator, &table, 2, 2);
    try std.testing.expectApproxEqRel(@as(f64, 50.0 / 63.0), r.statistic, 1e-13);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), r.df, 0);
    try std.testing.expectApproxEqRel(@as(f64, 0.3729984836134872), r.p, 1e-12);
}

test "contingencyChi2 — zero marginal is an error" {
    const allocator = std.testing.allocator;
    const table = [_]f64{ 0, 0, 30, 40 };
    try std.testing.expectError(error.ZeroMarginal, contingencyChi2(allocator, &table, 2, 2));
}
