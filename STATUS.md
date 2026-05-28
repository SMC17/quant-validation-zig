# STATUS

**Version:** v0.1.0
**Last update:** 2026-05-27
**Zig:** 0.16.0
**License:** AGPL-3.0

## Tests

30/30 passing under `zig build test` (Debug).

Breakdown:
- `src/stats.zig` — 6 internal tests (moments, normCdf reference points,
  normPpf round-trip + standard quantiles)
- `src/sharpe.zig` — 5 internal tests (PSR zero-SR identity, fat-tail
  penalty monotonicity, DSR null-max identity, expectedMaxSharpe
  reference number)
- `src/purged_cv.zig` — 3 internal tests (non-overlap reduction,
  overlap purging, embargo)
- `src/cpcv.zig` — 6 internal tests (C(K,n_test_groups) split count,
  holdout size, train/test disjointness, purging against a hand-
  constructed horizon pattern, reduction to purged K-Fold when
  n_test_groups=1, per-block embargo on non-adjacent test groups)
- `tests/test_reference_numbers.zig` — 9 external integration tests
  consuming the public module surface

## What's here

- `stats` — moments + `normCdf` (NR3 Chebyshev erfc) + `normPpf` (Acklam)
- `sharpe` — Sharpe ratio, PSR (Bailey & López de Prado 2012), DSR
  (Bailey & López de Prado 2014), expectedMaxSharpe
- `purged_cv` — Purged K-Fold with embargo (López de Prado AFML §7.4)
- `cpcv` — Combinatorial Purged Cross-Validation (López de Prado
  AFML §7.5), reusing the purging + embargo machinery from
  `purged_cv` with per-block embargo on non-adjacent test groups
- `examples/dsr_collapse.zig` — worked example wired as
  `zig build dsr-demo`; 100 noise-only backtests demonstrate the
  selection-bias collapse: naive PSR(0) approaches 1 while DSR
  collapses to roughly 0.5

## Precision

- `normCdf` uses the 28-term Numerical Recipes Chebyshev `erfc`
  approximation. Absolute error ~3e-8 against scipy.stats.norm.cdf
  in the discriminating mid-range. Sufficient for PSR / DSR work
  where input SR estimates carry orders of magnitude more uncertainty.
- `normPpf` uses Acklam's algorithm with ~1.15e-9 max relative error.
- All tests pass with tolerances tightened to the precision floor.

## What's open

### v0.2 substrate gaps
- Multiple-testing correction primitives (BH-FDR, Bonferroni)
- Hidden Markov regime detection (Baum-Welch, Viterbi)
- Minimum Backtest Length calculator
- Higher-precision `erfc` (Cody rational approximation, ~1e-15)
- Bootstrap + block-bootstrap confidence intervals
- Bayesian Sharpe estimation

## What this library is NOT

- Not a backtester. There is no execution model, no order book, no
  slippage. The caller supplies returns; this library helps you decide
  whether the returns are real.
- Not annualised. Multiply by `sqrt(periods_per_year)` at the caller.
- Not benchmarked against alternative implementations (e.g. mlfinlab,
  Riskfolio-Lib). Cross-validated against scipy primitives only.
