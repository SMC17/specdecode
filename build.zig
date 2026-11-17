const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "specdecode",
        .root_module = root_mod,
    });
    b.installArtifact(exe);

    const capi_mod = b.createModule(.{
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = optimize,
    });
    const lib = b.addLibrary(.{
        .name = "specdecode",
        .root_module = capi_mod,
        .linkage = .dynamic,
    });
    b.installArtifact(lib);

    const tests = b.addTest(.{ .root_module = root_mod });
    const capi_tests = b.addTest(.{ .root_module = capi_mod });
    const test_step = b.step("test", "Run all specdecode and C ABI tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
    test_step.dependOn(&b.addRunArtifact(capi_tests).step);
}
