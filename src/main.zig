// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! zstrata: measure every layer of this machine (CPU, RAM, the CPU-GPU link, the GPU) with the
//! same method, then run the same tasks on each to see where each kind of work belongs.
//!
//!     zig build run -Doptimize=ReleaseFast -- all > docs/data/strata.csv
//!     zig build run -Doptimize=ReleaseFast -- cpu --only alu,chase
//!     zig build run -Doptimize=ReleaseFast -- gpu --list-stats
//!
//! Output: one CSV (report.header) on stdout, progress on stderr.

const std = @import("std");
const builtin = @import("builtin");
const report = @import("report.zig");
const gpu = @import("gpu.zig");
const cpu_probes = @import("cpu/probes.zig");
const host = @import("cpu/host.zig");
const gpu_probes = @import("gpu/probes.zig");
const link_probes = @import("link/probes.zig");
const cross_probes = @import("cross/probes.zig");

pub const Part = enum { cpu, link, gpu, cross, all };

fn names(part: Part) []const []const u8 {
    return switch (part) {
        .cpu => &cpu_probes.all,
        .gpu => &gpu_probes.all,
        .link => &link_probes.all,
        .cross => &cross_probes.all,
        else => &.{},
    };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);

    var parts: std.ArrayList(Part) = .empty;
    var only: ?[]const []const u8 = null;
    var list_stats = false;
    var device_info = false;
    var no_check = false;
    var no_throttle = false;
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const a = argv[i];
        if (std.mem.eql(u8, a, "--only") and i + 1 < argv.len) {
            i += 1;
            var list: std.ArrayList([]const u8) = .empty;
            var it = std.mem.splitScalar(u8, argv[i], ',');
            while (it.next()) |n| try list.append(arena, n);
            only = list.items;
        } else if (std.mem.eql(u8, a, "--list-stats")) {
            list_stats = true;
        } else if (std.mem.eql(u8, a, "--device-info")) {
            device_info = true;
        } else if (std.mem.eql(u8, a, "--no-check")) {
            no_check = true;
        } else if (std.mem.eql(u8, a, "--no-throttle")) {
            no_throttle = true;
        } else if (std.meta.stringToEnum(Part, a)) |p| {
            if (p == .all) {
                try parts.appendSlice(arena, &.{ .cpu, .link, .gpu, .cross });
            } else try parts.append(arena, p);
        } else {
            std.log.err("unknown argument: {s} (parts: cpu link gpu cross all; --only a,b; --no-check; --no-throttle; --list-stats; --device-info)", .{a});
            std.process.exit(2);
        }
    }
    if (parts.items.len == 0) try parts.appendSlice(arena, &.{ .cpu, .link, .gpu, .cross });
    const quick = builtin.mode == .debug;
    if (quick) report.note("Debug build: small sizes, one run each (use -Doptimize=ReleaseFast to measure)\n", .{});

    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buf);
    const out = &stdout.interface;
    cpu_probes.initTopology(io);

    // Open the GPU only when a part needs it, so the CPU map still runs on a machine whose
    // Vulkan driver is missing or broken.
    const needs_gpu = list_stats or device_info or for (parts.items) |p| {
        if (p != .cpu) break true;
    } else false;
    var feats: gpu_probes.Features = .{};
    var ctx: ?gpu.Context = if (needs_gpu) try gpu.Context.init(gpa, feats.options(quick)) else null;
    defer if (ctx) |*c| c.deinit();
    const device = if (ctx) |*c| c.deviceName() else "not opened (cpu only)";
    report.note("gpu: {s}; cpu threads: {d}\n", .{ device, cpu_probes.cpuCount() });
    var lab: ?gpu_probes.Lab = if (ctx) |*c| try gpu_probes.Lab.init(c, quick) else null;
    defer if (lab) |*l| l.deinit();

    if (list_stats) return gpu_probes.listStats(&lab.?);
    if (device_info) {
        try ctx.?.writeDeviceInfo(gpa, out);
        return out.flush();
    }

    try out.writeAll(report.header);
    // --no-throttle: opt this process out of Windows power throttling (host.noThrottling). The
    // row says whether it took effect, so a run file records how it was made.
    if (no_throttle) {
        const ok = host.noThrottling();
        if (!ok) report.note("--no-throttle: not available here (Windows only) or the call failed\n", .{});
        try report.write(out, .{ .device = "cpu", .probe = "setting", .variant = "no-throttle", .x = 0, .value = if (ok) 1 else 0, .unit = "flag", .ns = 0 });
    }
    // Disturbance check on full runs: markers before each part and at the end (`cpu,check`
    // rows). Spot checks with --only skip it; they are not meant to be kept.
    var check: ?cpu_probes.Check = if (only == null and !no_check) try cpu_probes.Check.init(gpa, quick) else null;
    defer if (check) |*c| c.deinit(gpa);
    var point: u32 = 0;
    for (parts.items) |part| {
        if (check) |*c| {
            try c.take(gpa, out, point);
            point += 1;
        }
        for (only orelse names(part)) |name| {
            const known = switch (part) {
                .cpu => try cpu_probes.run(gpa, out, name, quick),
                .gpu => try gpu_probes.run(gpa, &lab.?, out, name),
                .link => try link_probes.run(gpa, &lab.?, out, name, quick),
                .cross => try cross_probes.run(gpa, &lab.?, out, name, quick),
                else => true,
            };
            if (!known) report.note("{t}: no probe named {s}, skipped\n", .{ part, name });
        }
    }
    if (check) |*c| {
        try c.take(gpa, out, point);
        _ = try c.finish(out);
    }

    const seen = gpu.validation_messages.load(.monotonic);
    if (seen != 0) {
        std.log.err("{d} validation message(s)", .{seen});
        std.process.exit(1);
    }
}

test {
    _ = cpu_probes;
    _ = gpu_probes;
    _ = link_probes;
    _ = cross_probes;
}
