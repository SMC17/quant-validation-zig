const std = @import("std");
const qv = @import("quant_validation");

// Worked example — reproduce a legacy research repo's chi-square claim
// in pure Zig.
//
// Source (read-only): ~/0theta-org/ZigLinguistics. Its
// analysis/statistical_summary.json documents, for "experiment 2"
// (Brown Corpus genre analysis), a genre × suffix chi-square with
//   chi_square_p_value = 5.081599549999679e-218
// The generating script (analysis/statistical_validation.py) builds the
// contingency table from results/experiment_2_genre_analysis/
// all_suffixes.csv: counts for the top-5 suffixes {s, ed, ing, tion, ly}
// pivoted across all 15 Brown Corpus genres, then calls
// scipy.stats.chi2_contingency → 15×5 table, df = (15-1)(5-1) = 56.
//
// The same JSON also documents anova_p_value = 8.65e-5. That number is
// a scipy.stats.f_oneway ANOVA on *percentage* coverage of -tion across
// narrative vs academic genre groups — an F-test, not a chi-square —
// so it is OUT OF SCOPE for this module and deliberately not reproduced.
//
// The table below is transcribed verbatim from the committed CSV
// (column order ed, ing, ly, s, tion — pandas pivot_table sorts
// columns lexicographically; row order is the 15 genres sorted, as
// pandas sorts the index). Row/column order does not affect X².

const documented_p: f64 = 5.081599549999679e-218;

const n_genres: usize = 15;
const n_suffixes: usize = 5;

const genres = [n_genres][]const u8{
    "adventure",       "belles_lettres", "editorial", "fiction",
    "government",      "hobbies",        "humor",     "learned",
    "lore",            "mystery",        "news",      "religion",
    "reviews",         "romance",        "science_fiction",
};

// zig fmt: off
const counts = [n_genres * n_suffixes]f64{
    // ed,  ing,  ly,   s,    tion
    1186,  729, 378, 1183, 107, // adventure
    1642, 1044, 600, 3369, 409, // belles_lettres
     818,  599, 329, 1738, 225, // editorial
    1161,  676, 348, 1424, 130, // fiction
     642,  498, 254, 1398, 247, // government
     895,  748, 301, 2140, 230, // hobbies
     473,  314, 195,  805,  75, // humor
    1339, 1002, 584, 2987, 508, // learned
    1328,  907, 476, 2596, 299, // lore
     846,  499, 301,  884,  91, // mystery
    1114,  761, 324, 2215, 257, // news
     573,  362, 263, 1126, 178, // religion
     620,  439, 312, 1583, 150, // reviews
     997,  618, 315, 1216, 112, // romance
     372,  186, 128,  456,  60, // science_fiction
};
// zig fmt: on

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();

    const r = try qv.chi2.contingencyChi2(allocator, &counts, n_genres, n_suffixes);

    const log10_p = @log(r.p) / std.math.ln10;
    const log10_doc = @log(documented_p) / std.math.ln10;
    const rel_err = @abs(r.p - documented_p) / documented_p;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);
    const w = &stdout_writer.interface;

    try w.print("ZigLinguistics chi-square reproduction — genre x suffix independence\n", .{});
    try w.print("  table                 = {d} genres x {d} suffixes (s, ed, ing, tion, ly)\n", .{ n_genres, n_suffixes });
    try w.print("  source                = results/experiment_2_genre_analysis/all_suffixes.csv\n", .{});
    try w.print("  first genre           = {s}, last genre = {s}\n", .{ genres[0], genres[n_genres - 1] });
    try w.print("  chi-square statistic  = {d:.10}\n", .{r.statistic});
    try w.print("  degrees of freedom    = {d}\n", .{r.df});
    try w.print("  p (this library)      = {e}\n", .{r.p});
    try w.print("  p (documented, scipy) = {e}\n", .{documented_p});
    try w.print("  log10 p (ours/theirs) = {d:.6} / {d:.6}\n", .{ log10_p, log10_doc });
    try w.print("  relative error        = {e}\n", .{rel_err});
    try w.print("\n", .{});
    try w.print("Second documented value (anova_p_value = 8.65e-5) is a scipy\n", .{});
    try w.print("f_oneway ANOVA on -tion percentage coverage, narrative vs academic\n", .{});
    try w.print("genre groups — an F-test, out of scope for the chi2 module.\n", .{});
    try w.flush();

    // The reproduction claim is load-bearing: fail loudly if the
    // recomputed p drifts from the documented scipy value by more than
    // a hair (1e-9 relative is far tighter than order-of-magnitude).
    if (rel_err > 1e-9) return error.ReproductionMismatch;
}
