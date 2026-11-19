// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

const std = @import("std");

const shaders = [_][]const u8{ "fma32", "fma16", "lds_bw", "stream", "chase", "coop_f16", "coop_bf16", "coop_f8", "coop_i8", "mlp_reg", "mlp_lds", "mlp_lds4", "mlp_mem", "mlp_mem4", "saxpy", "trees", "chains" };
const shader_includes = [_][]const u8{ "fma.glsl", "coop.glsl", "mlp.glsl" };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The library other projects import as `zstrata` (src/lib.zig): the thread pool, the vector
    // helpers and the radix sorts. It needs nothing but std.
    const lib = b.addModule("zstrata", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    const lib_tests = b.addRunArtifact(b.addTest(.{ .root_module = lib }));
    b.step("test-lib", "Run the library's tests (no GPU needed)").dependOn(&lib_tests.step);

    // A project that only imports the library passes this, so the measuring program, and with
    // it the Vulkan packages and the shader compiler, are never touched.
    const lib_only = b.option(bool, "lib-only", "Only the library module: skip the measuring program") orelse false;
    if (lib_only) return;

    // Null while the Vulkan packages are still being fetched; the build then runs again.
    const exe_mod = rootModule(b, target, optimize, false) orelse return;
    const exe = b.addExecutable(.{ .name = "zstrata", .root_module = exe_mod });
    b.installArtifact(exe);
    const run = b.addRunArtifact(exe);
    useLocalVulkan(b, run);
    run.addPassthruArgs();
    b.step("run", "Measure the machine (parts: cpu link gpu cross all)").dependOn(&run.step);

    // The page: docs/page/strata.html from its source, the chart helpers and the CSV.
    const page = b.addRunArtifact(b.addExecutable(.{ .name = "page", .root_module = b.createModule(.{
        .root_source_file = b.path("tools/page.zig"),
        .target = target,
        .optimize = optimize,
    }) }));
    page.setCwd(b.path("."));
    b.step("page", "Rebuild docs/page/strata.html from docs/data/strata.csv").dependOn(&page.step);

    // Repeatability: per-measurement median, range and spread over several run CSVs.
    const spread_mod = b.createModule(.{
        .root_source_file = b.path("tools/spread.zig"),
        .target = target,
        .optimize = optimize,
    });
    const spread = b.addRunArtifact(b.addExecutable(.{ .name = "spread", .root_module = spread_mod }));
    spread.setCwd(b.path("."));
    spread.addPassthruArgs();
    b.step("spread", "Compare run CSVs: zig build spread -- a.csv b.csv ...").dependOn(&spread.step);

    // Tests open the real GPU; there is no mock.
    const tests = b.addRunArtifact(b.addTest(.{ .root_module = exe_mod }));
    useLocalVulkan(b, tests);
    const test_step = b.step("test", "Run all tests (CPU and GPU)");
    test_step.dependOn(&tests.step);
    test_step.dependOn(&lib_tests.step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = spread_mod })).step);

    // ThreadSanitizer over the CPU-side tests (names start with "cpu"); the GPU tests stay out
    // because the Vulkan driver isn't instrumented. Needs LLVM, libc and an explicit
    // linux-gnu target.
    const tsan_target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu });
    const filter = b.option([]const u8, "test-filter", "Only tests whose name contains this (test-tsan: default \"cpu\")");
    const tsan_tests = b.addTest(.{ .root_module = rootModule(b, tsan_target, optimize, true) orelse return, .filters = &.{filter orelse "cpu"}, .use_llvm = true });
    const tsan_run = b.addRunArtifact(tsan_tests);
    useLocalVulkan(b, tsan_run);
    b.step("test-tsan", "Run the CPU-side (threaded) tests under ThreadSanitizer").dependOn(&tsan_run.step);
}

fn rootModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, tsan: bool) ?*std.Build.Module {
    const headers = b.lazyDependency("vulkan_headers", .{}) orelse return null;
    const vulkan = (b.lazyDependency("vulkan", .{
        .registry = headers.path("registry/vk.xml"),
    }) orelse return null).module("vulkan-zig");
    // libc so std.DynLib uses the system dlopen: the loader and the RADV driver it pulls in are
    // ordinary glibc shared objects.
    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .sanitize_thread = if (tsan) true else null,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan },
        },
    });
    for (shaders) |name| {
        mod.addAnonymousImport(b.fmt("{s}.spv", .{name}), .{ .root_source_file = compileShader(b, name) });
    }
    return mod;
}

/// shaders/<name>.comp -> SPIR-V with the persisted glslc, importable as "<name>.spv".
fn compileShader(b: *std.Build, name: []const u8) std.Build.LazyPath {
    const prefix = vulkanPrefix(b) orelse @panic("set HOME (USERPROFILE on Windows) or VKPREFIX to find glslc");
    const exe = if (hostIsWindows(b)) "glslc.exe" else "glslc";
    const glslc = b.addSystemCommand(&.{ b.fmt("{s}/bin/{s}", .{ prefix, exe }), "--target-env=vulkan1.3", "-O", "-o" });
    const spv = glslc.addOutputFileArg(b.fmt("{s}.spv", .{name}));
    glslc.addFileArg(b.path(b.fmt("shaders/{s}.comp", .{name})));
    for (shader_includes) |inc| glslc.addFileInput(b.path(b.fmt("shaders/{s}", .{inc})));
    if (!hostIsWindows(b)) glslc.setEnvironmentVariable("LD_LIBRARY_PATH", b.fmt("{s}/lib64", .{prefix}));
    return spv;
}

fn hostIsWindows(b: *std.Build) bool {
    return b.graph.host.result.os.tag == .windows;
}

/// Where glslc lives: `$VKPREFIX`, else the persisted Vulkan stack `~/.local/vulkan/usr` on Linux,
/// or the shaderc junction `%USERPROFILE%\.local\shaderc` on Windows (the driver ships the loader).
fn vulkanPrefix(b: *std.Build) ?[]const u8 {
    const env = &b.graph.environ_map;
    if (env.get("VKPREFIX")) |p| return p;
    if (hostIsWindows(b)) return b.fmt("{s}/.local/shaderc", .{env.get("USERPROFILE") orelse env.get("HOME") orelse return null});
    return b.fmt("{s}/.local/vulkan/usr", .{env.get("HOME") orelse return null});
}

/// Point a run step at the Vulkan stack persisted under ~/.local/vulkan (same variables as
/// tools/env.sh). VKPREFIX overrides the location. Windows needs only the validation layer's
/// folder (`%USERPROFILE%\.local\vk-layers`, `$VKLAYERS` overrides): the AMD driver installs
/// vulkan-1.dll and registers its own ICD.
fn useLocalVulkan(b: *std.Build, run: *std.Build.Step.Run) void {
    const env = &b.graph.environ_map;
    if (hostIsWindows(b)) {
        const layers = env.get("VKLAYERS") orelse b.fmt("{s}/.local/vk-layers", .{env.get("USERPROFILE") orelse return});
        run.setEnvironmentVariable("VK_ADD_LAYER_PATH", layers);
        return;
    }
    const prefix = vulkanPrefix(b) orelse return;
    const lib = b.fmt("{s}/lib64", .{prefix});
    run.setEnvironmentVariable("LD_LIBRARY_PATH", if (env.get("LD_LIBRARY_PATH")) |old| b.fmt("{s}:{s}", .{ lib, old }) else lib);
    run.setEnvironmentVariable("VK_DRIVER_FILES", b.fmt("{s}/share/vulkan/icd.d/radeon_icd.x86_64.json", .{prefix}));
    run.setEnvironmentVariable("VK_ADD_LAYER_PATH", b.fmt("{s}/share/vulkan/explicit_layer.d", .{prefix}));
    run.setEnvironmentVariable("MESA_VK_IGNORE_CONFORMANCE_WARNING", "1");
    run.setEnvironmentVariable("AMDGPU_ASIC_ID_TABLE_PATHS", b.fmt("{s}/share/libdrm", .{prefix}));
}
