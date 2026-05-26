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
}
