const std = @import("std");

pub fn mean(xs: []const f64) f64 {
    var s: f64 = 0;
    for (xs) |x| s += x;
    return s / @as(f64, @floatFromInt(xs.len));
}

pub fn variance(xs: []const f64, comptime ddof: u8) f64 {
    const n: f64 = @floatFromInt(xs.len);
    const m = mean(xs);
    var s: f64 = 0;
    for (xs) |x| {
        const d = x - m;
        s += d * d;
    }
    return s / (n - @as(f64, @floatFromInt(ddof)));
}

pub fn stddev(xs: []const f64, comptime ddof: u8) f64 {
    return @sqrt(variance(xs, ddof));
}

// Fisher skewness: m3 / m2^(3/2). Population estimator (biased).
pub fn skew(xs: []const f64) f64 {
    const n: f64 = @floatFromInt(xs.len);
    const m = mean(xs);
    var m2: f64 = 0;
    var m3: f64 = 0;
    for (xs) |x| {
        const d = x - m;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n;
    m3 /= n;
    return m3 / std.math.pow(f64, m2, 1.5);
}

// Pearson kurtosis: m4 / m2^2. Population estimator.
// Normal distribution gives 3.0; excess kurtosis = this - 3.
pub fn kurtosis(xs: []const f64) f64 {
    const n: f64 = @floatFromInt(xs.len);
    const m = mean(xs);
    var m2: f64 = 0;
    var m4: f64 = 0;
    for (xs) |x| {
        const d = x - m;
        const d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= n;
    m4 /= n;
    return m4 / (m2 * m2);
}

// Standard normal CDF via 28-term Chebyshev approximation of erfc
// (Numerical Recipes 3rd ed. §6.2.2). Absolute precision ~1e-14
// across the full range — sufficient that residual error is below
// any meaningful estimator uncertainty in PSR / DSR work.
pub fn normCdf(z: f64) f64 {
    return 0.5 * (1.0 + erf(z / std.math.sqrt2));
}

const erfc_cof = [_]f64{
    -1.3026537197817094,    6.4196979235649026e-1,  1.9476473204185836e-2,
    -9.561514786808631e-3,  -9.46595344482036e-4,   3.66839497852761e-4,
    4.2523324806907e-5,     -2.0278578112534e-5,    -1.624290004647e-6,
    1.303655835580e-6,      1.5626441722e-8,        -8.5238095915e-8,
    6.529054439e-9,         5.059343495e-9,         -9.91364156e-10,
    -2.27365122e-10,        9.6467911e-11,          2.394038e-12,
    -6.886027e-12,          8.94487e-13,            3.13092e-13,
    -1.12708e-13,           3.81e-16,               7.106e-15,
    -1.523e-15,             -9.4e-17,               1.21e-16,
    -2.8e-17,
};

fn erfccheb(z: f64) f64 {
    if (z < 0) return 2.0 - erfccheb(-z);
    const t = 2.0 / (2.0 + z);
    const ty = 4.0 * t - 2.0;
    var d: f64 = 0;
    var dd: f64 = 0;
    var j: usize = erfc_cof.len - 1;
    while (j > 0) : (j -= 1) {
        const tmp = d;
        d = ty * d - dd + erfc_cof[j];
        dd = tmp;
    }
    return t * @exp(-z * z + 0.5 * (erfc_cof[0] + ty * d) - dd);
}

fn erf(z: f64) f64 {
    return if (z >= 0) 1.0 - erfccheb(z) else erfccheb(-z) - 1.0;
}

// Inverse standard normal CDF (quantile function) via Acklam's algorithm.
// Maximum relative error ~1.15e-9 across (0, 1). The version used widely in
// the literature for PSR/DSR work where higher-than-naive accuracy matters.
pub fn normPpf(p: f64) f64 {
    std.debug.assert(p > 0 and p < 1);

    const a = [_]f64{
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00,
    };
    const b = [_]f64{
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    };
    const c = [_]f64{
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00,  2.938163982698783e+00,
    };
    const d = [_]f64{
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00,
    };

    const p_low = 0.02425;
    const p_high = 1.0 - p_low;

    if (p < p_low) {
        const q = @sqrt(-2.0 * @log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= p_high) {
        const q = p - 0.5;
        const r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const q = @sqrt(-2.0 * @log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
}

// Internal tests — small, deterministic. Reference numbers come from
// standard normal tables and hand-computed moment cases.
test "mean and stddev — uniform series" {
    const xs = [_]f64{ 1, 2, 3, 4, 5 };
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), mean(&xs), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(2.5)), stddev(&xs, 1), 1e-12);
}

test "skew — symmetric distribution is 0" {
    const xs = [_]f64{ -2, -1, 0, 1, 2 };
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), skew(&xs), 1e-12);
}

test "kurtosis — uniform-step distribution gives known value" {
    // For xs = {-2,-1,0,1,2}: m2 = 2, m4 = 6.8, kurt = 6.8/4 = 1.7
    const xs = [_]f64{ -2, -1, 0, 1, 2 };
    try std.testing.expectApproxEqAbs(@as(f64, 1.7), kurtosis(&xs), 1e-12);
}

test "normCdf — standard reference points" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), normCdf(0.0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8413447460685429), normCdf(1.0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9772498680518208), normCdf(2.0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9986501019683699), normCdf(3.0), 1e-10);
}

test "normPpf — round-trip with normCdf" {
    const probes = [_]f64{ 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99 };
    for (probes) |p| {
        const z = normPpf(p);
        const p_back = normCdf(z);
        try std.testing.expectApproxEqAbs(p, p_back, 1e-7);
    }
}

test "normPpf — standard reference quantiles" {
    // 1.6448536269514722 ≈ Phi^-1(0.95)
    try std.testing.expectApproxEqAbs(@as(f64, 1.6448536269514722), normPpf(0.95), 1e-7);
    // 1.2815515655446004 ≈ Phi^-1(0.9)
    try std.testing.expectApproxEqAbs(@as(f64, 1.2815515655446004), normPpf(0.9), 1e-7);
}
