# Changelog

## v0.1.0 — 2026-05-27

### Added
- `cpcv`: Combinatorial Purged Cross-Validation per López de Prado
  AFML §7.5. Generates `C(K, n_test_groups)` train/test splits over
  the same observation set, reusing the purging + embargo machinery
  from `purged_cv`. Adjacent test fold groups merge into single
  contiguous blocks before the embargo window is applied (single
  trailing embargo per merged block, not one per internal seam).
- `examples/dsr_collapse.zig` worked example wired as
  `zig build dsr-demo`: 100 simulated strategy backtests over
  N=252 days, all pure noise. Shows naive PSR(0) approaching 1
  versus DSR collapsing toward 0.5 once the best-of-100 threshold
  is re-imposed.
- GitHub Actions CI: `zig build test` + `zig build dsr-demo` on
  every push and PR to main, Zig 0.16.0 pinned.
- 7 new tests (6 internal CPCV + 1 external CPCV split-count).
  Test total: 22 → 30.

## v0.0.1 — 2026-05-26

Initial substrate. Three modules, 22 passing tests.

### Added
- `stats`: moments (mean, variance, stddev, skew, Pearson kurtosis),
  high-precision `normCdf` (NR3 Chebyshev erfc, ~3e-8 absolute error),
  `normPpf` (Acklam, ~1.15e-9 max relative error)
- `sharpe`: non-annualised Sharpe ratio, Probabilistic Sharpe Ratio
  (Bailey & López de Prado 2012), Deflated Sharpe Ratio
  (Bailey & López de Prado 2014), expected max-Sharpe under the null
- `purged_cv`: Purged K-Fold with embargo (López de Prado AFML §7.4)
- Reference-number tests cross-checked against scipy.stats primitives

### Known limits
- `erfc` precision floor is ~3e-8; sufficient for SR/PSR/DSR but a
  future v0.2 will swap in a Cody rational approximation for ~1e-15
- No CPCV, regime detection, or multiple-testing correction yet
  (planned for v0.1)
- No worked end-to-end example yet (planned for v0.1)
