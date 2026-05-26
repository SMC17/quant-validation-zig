# Changelog

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
