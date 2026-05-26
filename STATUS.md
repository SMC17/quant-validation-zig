# STATUS

**Version:** v0.0.1
**Last update:** 2026-05-26
**Zig:** 0.16.0
**License:** AGPL-3.0

## Tests

22/22 passing under `zig build test` (ReleaseSafe / Debug).

Breakdown:
- `src/stats.zig` — 6 internal tests (moments, normCdf reference points,
  normPpf round-trip + standard quantiles)
- `src/sharpe.zig` — 5 internal tests (PSR zero-SR identity, fat-tail
  penalty monotonicity, DSR null-max identity, expectedMaxSharpe
  reference number)
- `src/purged_cv.zig` — 3 internal tests (non-overlap reduction,
  overlap purging, embargo)
- `tests/test_reference_numbers.zig` — 8 external integration tests
  consuming the public module surface

## Precision

- `normCdf` uses the 28-term Numerical Recipes Chebyshev `erfc`
  approximation. Absolute error ~3e-8 against scipy.stats.norm.cdf
  in the discriminating mid-range. Sufficient for PSR / DSR work
  where input SR estimates carry orders of magnitude more uncertainty.
- `normPpf` uses Acklam's algorithm with ~1.15e-9 max relative error.
- All tests pass with tolerances tightened to the precision floor.

## What's open

### v0.1 substrate gaps (3-4 weeks)
- Combinatorial Purged Cross-Validation (CPCV) — AFML §7.5
- Multiple-testing correction primitives (BH-FDR, Bonferroni)
- Hidden Markov regime detection (Baum-Welch, Viterbi)
- Worked example: apply the suite to a public-domain returns series
  and demonstrate the deflated-Sharpe collapse

### v0.2 substrate gaps
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
