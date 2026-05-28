// quant-validation-zig — statistical-validation primitives for quant research.
//
// Public surface. The library is organised into four modules:
//   - stats:     moment estimators + normal CDF / inverse CDF
//   - sharpe:    Sharpe ratio, Probabilistic Sharpe Ratio, Deflated Sharpe Ratio
//   - purged_cv: Purged K-Fold cross-validation generator (López de Prado AFML §7.4)
//   - cpcv:      Combinatorial Purged Cross-Validation (López de Prado AFML §7.5)
//
// Pre-1.0 substrate. Vocabulary deliberately matches the evidence:
// the implementations are tested against published reference numbers,
// not against a production backtesting harness — see STATUS.md.

pub const stats = @import("stats.zig");
pub const sharpe = @import("sharpe.zig");
pub const purged_cv = @import("purged_cv.zig");
pub const cpcv = @import("cpcv.zig");

test {
    _ = stats;
    _ = sharpe;
    _ = purged_cv;
    _ = cpcv;
}
