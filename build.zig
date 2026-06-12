const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("quant_validation", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tests = b.addTest(.{
        .root_module = mod,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // External integration tests against the public API as a consumer.
    const ext_tests_mod = b.createModule(.{
        .root_source_file = b.path("tests/test_reference_numbers.zig"),
        .target = target,
        .optimize = optimize,
    });
    ext_tests_mod.addImport("quant_validation", mod);
    const ext_tests = b.addTest(.{
        .root_module = ext_tests_mod,
    });
    const run_ext_tests = b.addRunArtifact(ext_tests);
    test_step.dependOn(&run_ext_tests.step);

    // Worked example — `zig build dsr-demo` runs the DSR collapse
    // demonstration: 100 noise-only backtests, naive PSR(0) vs DSR.
    const dsr_demo_mod = b.createModule(.{
        .root_source_file = b.path("examples/dsr_collapse.zig"),
        .target = target,
        .optimize = optimize,
    });
    dsr_demo_mod.addImport("quant_validation", mod);
    const dsr_demo_exe = b.addExecutable(.{
        .name = "dsr_collapse",
        .root_module = dsr_demo_mod,
    });
    const run_dsr_demo = b.addRunArtifact(dsr_demo_exe);
    const dsr_demo_step = b.step("dsr-demo", "Run the DSR collapse worked example");
    dsr_demo_step.dependOn(&run_dsr_demo.step);

    // Worked example — `zig build repro-lingstats` recomputes the
    // ZigLinguistics genre × suffix chi-square (documented p ≈ 5.08e-218)
    // in pure Zig from the committed contingency counts.
    const repro_mod = b.createModule(.{
        .root_source_file = b.path("examples/repro_lingstats.zig"),
        .target = target,
        .optimize = optimize,
    });
    repro_mod.addImport("quant_validation", mod);
    const repro_exe = b.addExecutable(.{
        .name = "repro_lingstats",
        .root_module = repro_mod,
    });
    const run_repro = b.addRunArtifact(repro_exe);
    const repro_step = b.step("repro-lingstats", "Reproduce the ZigLinguistics chi-square claim");
    repro_step.dependOn(&run_repro.step);
}
