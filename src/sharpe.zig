const std = @import("std");
const stats = @import("stats.zig");

// All Sharpe ratios in this module are NON-ANNUALIZED (per-observation).
// Annualisation is left to the caller — multiply by sqrt(periods_per_year).

pub fn sharpeRatio(returns: []const f64, risk_free_per_period: f64) f64 {
    const m = stats.mean(returns);
    const s = stats.stddev(returns, 1);
    return (m - risk_free_per_period) / s;
}

// Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).
//
// PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(n - 1)
//                 / sqrt(1 - skew*SR_hat + ((kurt - 1)/4) * SR_hat^2) )
//
// where SR_hat is the observed non-annualised Sharpe, skew/kurt are
// moments of the returns (Pearson kurtosis, normal = 3), and SR* is
// the benchmark Sharpe (often 0).
pub fn psr(
    sr_hat: f64,
    sr_benchmark: f64,
    n: usize,
    skewness: f64,
    kurt: f64,
) f64 {
    std.debug.assert(n >= 2);
    const n_f: f64 = @floatFromInt(n);
    const denom_sq = 1.0 - skewness * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat * sr_hat;
    std.debug.assert(denom_sq > 0);
    const z = (sr_hat - sr_benchmark) * @sqrt(n_f - 1.0) / @sqrt(denom_sq);
    return stats.normCdf(z);
}

// Convenience: compute PSR directly from a returns vector against a
// benchmark Sharpe. Pulls n / skew / kurt from the sample itself.
pub fn psrFromReturns(returns: []const f64, sr_benchmark: f64) f64 {
    const sr_hat = sharpeRatio(returns, 0);
    return psr(sr_hat, sr_benchmark, returns.len, stats.skew(returns), stats.kurtosis(returns));
}

// Expected maximum Sharpe under the null across N independent trials
// (Bailey & López de Prado 2014, "The Deflated Sharpe Ratio", eq. 7).
//
//   E[max SR_n] ≈ sqrt(V[SR_n]) * ( (1 - gamma) * Phi^-1(1 - 1/N)
//                                  + gamma     * Phi^-1(1 - 1/(N*e)) )
//
// gamma is Euler-Mascheroni. Used as the threshold SR* inside DSR.
pub fn expectedMaxSharpe(variance_of_trial_sharpes: f64, num_trials: usize) f64 {
    std.debug.assert(num_trials >= 2);
    const gamma: f64 = 0.5772156649015329;
    const n_f: f64 = @floatFromInt(num_trials);
    const e = std.math.e;
    const q1 = stats.normPpf(1.0 - 1.0 / n_f);
    const q2 = stats.normPpf(1.0 - 1.0 / (n_f * e));
    return @sqrt(variance_of_trial_sharpes) * ((1.0 - gamma) * q1 + gamma * q2);
}

// Deflated Sharpe Ratio (Bailey & López de Prado 2014).
//
// DSR = PSR( SR0 ) where SR0 = E[max SR_n] under the null.
// Returns the probability that the observed SR is genuinely above the
// null-maximum after correcting for selection bias from num_trials.
pub fn dsr(
    sr_hat: f64,
    n: usize,
    skewness: f64,
    kurt: f64,
    variance_of_trial_sharpes: f64,
    num_trials: usize,
) f64 {
    const sr0 = expectedMaxSharpe(variance_of_trial_sharpes, num_trials);
    return psr(sr_hat, sr0, n, skewness, kurt);
}

// ---- internal tests ----

test "PSR — zero observed SR equals 50 percent" {
    // SR_hat = 0 against SR* = 0 with normal returns → Phi(0) = 0.5
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), psr(0.0, 0.0, 100, 0.0, 3.0), 1e-12);
}

test "PSR — discriminating reference case" {
    // SR_hat = 0.1, n = 20, SR* = 0, normal (skew=0, kurt=3).
    //
    //   denom = sqrt(1 + 0 + ((3-1)/4) * 0.01) = sqrt(1.005) = 1.0024968827881711
    //   numer = 0.1 * sqrt(19)                                = 0.43588989435...
    //   z     = numer / denom                                 = 0.43480610...
    //   PSR   = Phi(z)                                        ≈ 0.6681477029766314
    //
    // Cross-checked against scipy.stats.norm.cdf; NR3 Chebyshev erfc
    // matches to ~3e-8 in this regime (1e-7 tolerance is safe).
    try std.testing.expectApproxEqAbs(@as(f64, 0.6681477029766314), psr(0.1, 0.0, 20, 0.0, 3.0), 1e-7);
}

test "PSR — high SR with negative skew is penalised" {
    // Negative skew + high kurt should INCREASE the denominator,
    // so PSR drops relative to the same SR_hat under normality.
    const psr_normal = psr(1.0, 0.0, 50, 0.0, 3.0);
    const psr_fat = psr(1.0, 0.0, 50, -0.5, 4.5);
    try std.testing.expect(psr_fat < psr_normal);
}

test "expectedMaxSharpe — N=2 baseline sanity" {
    // V = 1.0, N = 2 → SR0 = gamma * Phi^-1(1 - 1/(2e))
    //                       = 0.5772156649 * Phi^-1(0.81606028)
    //                       = 0.5772156649 * 0.9003817547
    //                       = 0.5197553440822881
    //
    // Cross-checked against scipy.stats.norm.ppf to ~1e-9.
    const sr0 = expectedMaxSharpe(1.0, 2);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5197553440822881), sr0, 1e-7);
}

test "DSR — observing the null-max gives 50 percent" {
    // Construct a case where SR_hat exactly equals SR0 → DSR = 0.5.
    const v: f64 = 0.04;
    const n_trials: usize = 100;
    const sr0 = expectedMaxSharpe(v, n_trials);
    const d = dsr(sr0, 250, 0.0, 3.0, v, n_trials);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), d, 1e-9);
}
