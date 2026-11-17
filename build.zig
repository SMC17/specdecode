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

    // WebAssembly freestanding target (`zig build wasm`)
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });
    const wasm_mod = b.createModule(.{
        .root_source_file = b.path("src/wasm.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    const wasm_exe = b.addExecutable(.{
        .name = "specdecode",
        .root_module = wasm_mod,
    });
    wasm_exe.entry = .disabled;
    wasm_exe.rdynamic = true;
    const wasm_step = b.step("wasm", "Build specdecode.wasm (wasm32-freestanding)");
    wasm_step.dependOn(&b.addInstallArtifact(wasm_exe, .{}).step);

    const tests = b.addTest(.{ .root_module = root_mod });
    const capi_tests = b.addTest(.{ .root_module = capi_mod });
    const test_step = b.step("test", "Run all specdecode and C ABI tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
    test_step.dependOn(&b.addRunArtifact(capi_tests).step);
}
