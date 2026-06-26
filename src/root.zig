// quant-validation-zig — statistical-validation primitives for quant research.
//
// Public surface. The library is organised into five modules:
//   - stats:     moment estimators + normal CDF / inverse CDF + erfc
//   - sharpe:    Sharpe ratio, Probabilistic Sharpe Ratio, Deflated Sharpe Ratio
//   - purged_cv: Purged K-Fold cross-validation generator (López de Prado AFML §7.4)
//   - cpcv:      Combinatorial Purged Cross-Validation (López de Prado AFML §7.5)
//   - chi2:      regularized incomplete gamma P/Q (NR3 §6.2), chi-square
//                survival function, Pearson contingency-table test
//
// Pre-1.0 substrate. Vocabulary deliberately matches the evidence:
// the implementations are tested against published reference numbers,
// not against a production backtesting harness — see STATUS.md.
//
//   - backtest:  CPCV path distribution — holdout scoring, PathDistribution
//               summary (mean/stddev/min/max/positive_fraction over all paths)

pub const stats = @import("stats.zig");
pub const sharpe = @import("sharpe.zig");
pub const purged_cv = @import("purged_cv.zig");
pub const cpcv = @import("cpcv.zig");
pub const chi2 = @import("chi2.zig");
pub const backtest = @import("backtest.zig");

test {
    _ = stats;
    _ = sharpe;
    _ = purged_cv;
    _ = cpcv;
    _ = chi2;
    _ = backtest;
}
