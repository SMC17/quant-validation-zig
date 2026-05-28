const std = @import("std");
const qv = @import("quant_validation");

// Worked example — Deflated Sharpe Ratio collapse demonstration.
//
// Setup: simulate 100 strategy backtests over one trading year (N=252).
// Every series is centred Normal(0, 0.01) — pure noise, TRUE Sharpe = 0
// for every trial. A naive analyst would pick the trial with the largest
// observed Sharpe and report its PSR(0); the Deflated Sharpe Ratio
// corrects for the selection bias and should collapse to roughly the null.
//
// Reproduces the AFML §8 / Bailey-López de Prado 2014 thesis on selection
// bias in published Sharpe ratios.

const NUM_TRIALS: usize = 100;
const SERIES_LEN: usize = 252;
const SEED: u64 = 17;
const SIGMA: f64 = 0.01;

// Box-Muller transform — turns two U(0,1) draws into one N(0,1) draw.
fn standardNormal(rng: std.Random) f64 {
    var uu: f64 = rng.float(f64);
    if (uu < 1e-300) uu = 1e-300; // guard against log(0)
    const uv: f64 = rng.float(f64);
    return @sqrt(-2.0 * @log(uu)) * @cos(2.0 * std.math.pi * uv);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();

    var prng: std.Random.DefaultPrng = .init(SEED);
    const rng = prng.random();

    // Allocate one returns slice per trial and one Sharpe scalar per trial.
    const trial_sharpes = try allocator.alloc(f64, NUM_TRIALS);

    var best_idx: usize = 0;
    var best_sr: f64 = -std.math.inf(f64);
    const best_returns = try allocator.alloc(f64, SERIES_LEN);

    const scratch = try allocator.alloc(f64, SERIES_LEN);

    for (0..NUM_TRIALS) |t| {
        for (0..SERIES_LEN) |i| {
            scratch[i] = SIGMA * standardNormal(rng);
        }
        const sr = qv.sharpe.sharpeRatio(scratch, 0.0);
        trial_sharpes[t] = sr;
        if (sr > best_sr) {
            best_sr = sr;
            best_idx = t;
            @memcpy(best_returns, scratch);
        }
    }

    // Naive PSR — the bias-blind report.
    const naive_psr = qv.sharpe.psrFromReturns(best_returns, 0.0);

    // Deflated Sharpe Ratio — correct for having tried NUM_TRIALS variants.
    const var_trial = qv.stats.variance(trial_sharpes, 1);
    const skew_best = qv.stats.skew(best_returns);
    const kurt_best = qv.stats.kurtosis(best_returns);
    const deflated = qv.sharpe.dsr(
        best_sr,
        SERIES_LEN,
        skew_best,
        kurt_best,
        var_trial,
        NUM_TRIALS,
    );

    // Threshold the naive analyst defeated; the value the DSR re-imposes.
    const sr0 = qv.sharpe.expectedMaxSharpe(var_trial, NUM_TRIALS);

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);
    const w = &stdout_writer.interface;

    try w.print("DSR collapse demo — {d} trials of pure noise, N={d}\n", .{ NUM_TRIALS, SERIES_LEN });
    try w.print("  seed                  = {d}\n", .{SEED});
    try w.print("  sigma                 = {d}\n", .{SIGMA});
    try w.print("  best trial index      = {d}\n", .{best_idx});
    try w.print("  best naive SR         = {d:.6}\n", .{best_sr});
    try w.print("  V[trial Sharpes]      = {d:.6}\n", .{var_trial});
    try w.print("  E[max SR | null]      = {d:.6}\n", .{sr0});
    try w.print("  naive PSR(0)          = {d:.6}\n", .{naive_psr});
    try w.print("  Deflated Sharpe Ratio = {d:.6}\n", .{deflated});
    try w.print("\n", .{});
    try w.print("Interpretation: naive PSR rejects the null with high confidence;\n", .{});
    try w.print("DSR collapses toward 0.5 because the best-of-{d} threshold is\n", .{NUM_TRIALS});
    try w.print("re-imposed. AFML §8 / Bailey-Lopez de Prado 2014.\n", .{});
    try w.flush();
}
