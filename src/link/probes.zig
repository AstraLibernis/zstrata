// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! The link between the CPU and the GPU: what it costs to hand data and work across.
//!
//!   upload     CPU memory -> VRAM by size: staged (copy into a CPU-side buffer, then the GPU
//!              copies it over), direct (the CPU writes straight into VRAM through the
//!              resizable BAR), and the GPU's copy alone (timestamps)
//!   download   VRAM -> CPU memory by size: the GPU copies into a cached CPU-side buffer, then
//!              the CPU copies it out; and the CPU reading VRAM directly (uncached)
//!   roundtrip  fixed cost of one trip: an empty submit, and a tiny dispatch whose 4-byte
//!              result is read back; then split: recording vs resubmitting, and sleeping in
//!              vkWaitForFences vs polling the fence vs polling briefly, then sleeping (hybrid)
//!   wait       a job of a few ms waited for each way: wall time and the waiting thread's CPU
//!   batch      many tiny dependent dispatches: one submit for all vs one submit each
//!
//! Times are wall-clock on the CPU (what a program waiting for the result sees), median of 5
//! (round trips: median of 200).

const std = @import("std");
const gpu = @import("../gpu.zig");
const vk = gpu.vk;
const report = @import("../report.zig");
const clock = @import("../cpu/probes.zig");
const host = @import("../cpu/host.zig");
const Lab = @import("../gpu/probes.zig").Lab;
const gpu_probes = @import("../gpu/probes.zig");

const reps = 5;

fn sizes(quick: bool) []const u64 {
    return if (quick) &.{ 4 << 10, 1 << 20 } else &.{ 4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20, 1 << 30 };
}

const usage: vk.BufferUsageFlags = .{ .storage_buffer = true, .transfer_dst = true, .transfer_src = true };

pub fn probeUpload(gpa: std.mem.Allocator, lab: *Lab, out: *std.Io.Writer, quick: bool) !void {
    const ctx = lab.ctx;
    const max = sizes(quick)[sizes(quick).len - 1];
    const src = try gpa.alloc(u8, max);
    defer gpa.free(src);
    for (src, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 13);
    var staging = try gpu.Buffer.init(ctx, max, .{ .transfer_src = true }, .{ .host_visible = true, .host_coherent = true }, .{});
    defer staging.deinit(ctx);
    var direct = try gpu.Buffer.init(ctx, max, usage, .{ .device_local = true, .host_visible = true, .host_coherent = true }, .{});
    defer direct.deinit(ctx);

    for (sizes(quick)) |s| {
        var staged: [reps]f64 = undefined;
        var dma: [reps]f64 = undefined;
        var straight: [reps]f64 = undefined;
        for (0..reps) |r| {
            // Staged: CPU copy into the staging buffer, GPU copy into VRAM, wait.
            var t0 = clock.nowNs();
            @memcpy(staging.mapped.?[0..s], src[0..s]);
            try lab.cmds.begin(ctx);
            lab.stamps.reset(ctx, lab.cmds.cmd);
            lab.stamps.write(ctx, lab.cmds.cmd, 0);
            ctx.vkd.cmdCopyBuffer(lab.cmds.cmd, staging.handle, lab.big.handle, &.{.{ .src_offset = 0, .dst_offset = 0, .size = s }});
            lab.stamps.write(ctx, lab.cmds.cmd, 1);
            try lab.cmds.submitAndWait(ctx);
            staged[r] = @floatFromInt(clock.nowNs() - t0);
            dma[r] = try lab.stamps.elapsedNs(ctx, 0, 1);
            // Direct: the CPU writes into VRAM itself.
            t0 = clock.nowNs();
            @memcpy(direct.mapped.?[0..s], src[0..s]);
            straight[r] = @floatFromInt(clock.nowNs() - t0);
        }
        const sz: f64 = @floatFromInt(s);
        inline for (.{ .{ "staged", &staged }, .{ "direct-write", &straight }, .{ "gpu-copy-only", &dma } }) |c| {
            const ns = report.median(c[1]);
            try report.write(out, .{ .device = "link", .probe = "upload", .variant = c[0], .x = sz, .value = sz / ns, .unit = "GB/s", .ns = ns });
        }
    }
}

pub fn probeDownload(gpa: std.mem.Allocator, lab: *Lab, out: *std.Io.Writer, quick: bool) !void {
    const ctx = lab.ctx;
    const max = sizes(quick)[sizes(quick).len - 1];
    const dst = try gpa.alloc(u8, max);
    defer gpa.free(dst);
    var readback = try gpu.Buffer.init(ctx, max, .{ .transfer_dst = true }, .{ .host_visible = true, .host_coherent = true }, .{ .host_cached = true });
    defer readback.deinit(ctx);
    var direct = try gpu.Buffer.init(ctx, max, usage, .{ .device_local = true, .host_visible = true, .host_coherent = true }, .{});
    defer direct.deinit(ctx);
    @memset(direct.mapped.?[0..max], 7);

    for (sizes(quick)) |s| {
        var staged: [reps]f64 = undefined;
        var straight: [reps]f64 = undefined;
        // Reading VRAM directly is uncached and very slow; stop at 16 MiB.
        const do_direct = s <= 16 << 20;
        for (0..reps) |r| {
            var t0 = clock.nowNs();
            try lab.cmds.begin(ctx);
            ctx.vkd.cmdCopyBuffer(lab.cmds.cmd, lab.big.handle, readback.handle, &.{.{ .src_offset = 0, .dst_offset = 0, .size = s }});
            gpu.barrier(ctx, lab.cmds.cmd, .{ .copy = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
            try lab.cmds.submitAndWait(ctx);
            @memcpy(dst[0..s], readback.mapped.?[0..s]);
            staged[r] = @floatFromInt(clock.nowNs() - t0);
            if (do_direct) {
                t0 = clock.nowNs();
                @memcpy(dst[0..s], direct.mapped.?[0..s]);
                straight[r] = @floatFromInt(clock.nowNs() - t0);
            }
        }
        std.mem.doNotOptimizeAway(dst[s - 1]);
        const sz: f64 = @floatFromInt(s);
        var ns = report.median(&staged);
        try report.write(out, .{ .device = "link", .probe = "download", .variant = "staged", .x = sz, .value = sz / ns, .unit = "GB/s", .ns = ns });
        if (do_direct) {
            ns = report.median(&straight);
            try report.write(out, .{ .device = "link", .probe = "download", .variant = "direct-read", .x = sz, .value = sz / ns, .unit = "GB/s", .ns = ns });
        }
    }
}

pub fn probeRoundtrip(gpa: std.mem.Allocator, lab: *Lab, out: *std.Io.Writer, quick: bool) !void {
    const ctx = lab.ctx;
    const rounds: usize = if (quick) 5 else 200;
    var readback = try gpu.Buffer.init(ctx, 4, .{ .transfer_dst = true }, .{ .host_visible = true, .host_coherent = true }, .{ .host_cached = true });
    defer readback.deinit(ctx);
    var pipe = try lab.pipeline(gpu_probes.fma32Code(), &.{lab.out.handle}, @sizeOf(gpu_probes.FmaPush), .{ .subgroup_size = 64, .spec = &.{ 1, 1 } });
    defer pipe.deinit(ctx);
    const push: gpu_probes.FmaPush = .{ .iters = 1, .lanes = 64, .a = 1, .b = 1 };

    const empty = try gpa.alloc(f64, rounds);
    defer gpa.free(empty);
    const tiny = try gpa.alloc(f64, rounds);
    defer gpa.free(tiny);
    for (0..rounds) |r| {
        var t0 = clock.nowNs();
        try lab.cmds.begin(ctx);
        try lab.cmds.submitAndWait(ctx);
        empty[r] = @floatFromInt(clock.nowNs() - t0);

        t0 = clock.nowNs();
        try lab.cmds.begin(ctx);
        pipe.dispatch(ctx, lab.cmds.cmd, &push, .{ 1, 1, 1 });
        gpu.barrier(ctx, lab.cmds.cmd, .{ .compute_shader = true }, .{ .shader_storage_write = true }, .{ .copy = true }, .{ .transfer_read = true });
        ctx.vkd.cmdCopyBuffer(lab.cmds.cmd, lab.out.handle, readback.handle, &.{.{ .src_offset = 0, .dst_offset = 0, .size = 4 }});
        gpu.barrier(ctx, lab.cmds.cmd, .{ .copy = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
        try lab.cmds.submitAndWait(ctx);
        std.mem.doNotOptimizeAway(readback.slice(f32)[0]);
        tiny[r] = @floatFromInt(clock.nowNs() - t0);
    }
    var ns = report.median(empty);
    try report.write(out, .{ .device = "link", .probe = "roundtrip", .variant = "empty-submit", .x = 0, .value = ns / 1e3, .unit = "us", .ns = ns });
    ns = report.median(tiny);
    try report.write(out, .{ .device = "link", .probe = "roundtrip", .variant = "tiny-dispatch-readback", .x = 4, .value = ns / 1e3, .unit = "us", .ns = ns });

    // Where an empty trip's time goes: recording (resubmit a buffer recorded once) and the
    // wake-up after the fence (poll it instead of sleeping in vkWaitForFences).
    var again = try gpu.Commands.init(ctx);
    defer again.deinit(ctx);
    try again.beginReusable(ctx);
    try ctx.vkd.endCommandBuffer(again.cmd);
    const Split = struct { name: []const u8, record: bool, wait: gpu.Commands.Wait };
    const splits = [_]Split{
        .{ .name = "empty-submit-spin", .record = true, .wait = .spin },
        .{ .name = "empty-resubmit", .record = false, .wait = .block },
        .{ .name = "empty-resubmit-spin", .record = false, .wait = .spin },
        .{ .name = "empty-resubmit-hybrid", .record = false, .wait = .hybrid },
    };
    for (splits) |sp| {
        for (0..rounds) |r| {
            const t0 = clock.nowNs();
            if (sp.record) {
                try lab.cmds.begin(ctx);
                try ctx.vkd.endCommandBuffer(lab.cmds.cmd);
                try lab.cmds.resubmit(ctx, sp.wait);
            } else try again.resubmit(ctx, sp.wait);
            empty[r] = @floatFromInt(clock.nowNs() - t0);
        }
        ns = report.median(empty);
        try report.write(out, .{ .device = "link", .probe = "roundtrip", .variant = sp.name, .x = 0, .value = ns / 1e3, .unit = "us", .ns = ns });
    }
}

/// What each way of waiting costs on a job long enough that the wait mode matters for the CPU:
/// wall time (job + wait) and the waiting thread's own CPU time.
pub fn probeWait(gpa: std.mem.Allocator, lab: *Lab, out: *std.Io.Writer, quick: bool) !void {
    const ctx = lab.ctx;
    const rounds: usize = if (quick) 3 else 40;
    var pipe = try lab.pipeline(gpu_probes.fma32Code(), &.{lab.out.handle}, @sizeOf(gpu_probes.FmaPush), .{ .subgroup_size = 64, .spec = &.{ 1, 1 } });
    defer pipe.deinit(ctx);
    // One workgroup of dependent FMAs: a few milliseconds on the GPU, nothing for the CPU to do.
    const push: gpu_probes.FmaPush = .{ .iters = if (quick) 1 << 10 else 1 << 18, .lanes = 64, .a = 0.999, .b = 0.001 };
    var cmds = try gpu.Commands.init(ctx);
    defer cmds.deinit(ctx);
    try cmds.beginReusable(ctx);
    pipe.dispatch(ctx, cmds.cmd, &push, .{ 1, 1, 1 });
    try ctx.vkd.endCommandBuffer(cmds.cmd);

    const wall = try gpa.alloc(f64, rounds);
    defer gpa.free(wall);
    const cpu = try gpa.alloc(f64, rounds);
    defer gpa.free(cpu);
    _ = host.threadCpuNs(); // the first call calibrates (Windows: ~10 ms spin); keep it out of the rows
    inline for (.{ .block, .spin, .hybrid }) |mode| {
        for (0..rounds) |r| {
            const t0 = clock.nowNs();
            const c0 = host.threadCpuNs();
            try cmds.resubmit(ctx, mode);
            cpu[r] = @floatFromInt(host.threadCpuNs() - c0);
            wall[r] = @floatFromInt(clock.nowNs() - t0);
        }
        // Mean CPU over the rounds (a wait's CPU cost adds up across calls).
        var cpu_sum: f64 = 0;
        for (cpu) |c| cpu_sum += c;
        const name = @tagName(mode);
        var buf: [32]u8 = undefined;
        var ns = report.median(wall);
        try report.write(out, .{ .device = "link", .probe = "wait", .variant = try std.mem.print(&buf, "long-{s}-wall", .{name}), .x = 0, .value = ns / 1e3, .unit = "us", .ns = ns });
        ns = cpu_sum / @as(f64, @floatFromInt(rounds));
        try report.write(out, .{ .device = "link", .probe = "wait", .variant = try std.mem.print(&buf, "long-{s}-cpu", .{name}), .x = 0, .value = ns / 1e3, .unit = "us", .ns = ns });
    }
}

/// Many small dependent dispatches: all in one submit (with a barrier between each) vs one
/// submit + wait per dispatch. us per dispatch, by batch size.
pub fn probeBatch(gpa: std.mem.Allocator, lab: *Lab, out: *std.Io.Writer, quick: bool) !void {
    const ctx = lab.ctx;
    const rounds: usize = if (quick) 2 else 20;
    var pipe = try lab.pipeline(gpu_probes.fma32Code(), &.{lab.out.handle}, @sizeOf(gpu_probes.FmaPush), .{ .subgroup_size = 64, .spec = &.{ 1, 1 } });
    defer pipe.deinit(ctx);
    const push: gpu_probes.FmaPush = .{ .iters = 1, .lanes = 64, .a = 1, .b = 1 };
    const ks: []const u32 = if (quick) &.{ 1, 4 } else &.{ 1, 4, 16, 64, 256 };
    const one = try gpa.alloc(f64, rounds);
    defer gpa.free(one);
    const each = try gpa.alloc(f64, rounds);
    defer gpa.free(each);
    for (ks) |k| {
        for (0..rounds) |r| {
            var t0 = clock.nowNs();
            try lab.cmds.begin(ctx);
            for (0..k) |i| {
                if (i > 0) gpu.barrier(ctx, lab.cmds.cmd, .{ .compute_shader = true }, .{ .shader_storage_write = true }, .{ .compute_shader = true }, .{ .shader_storage_read = true, .shader_storage_write = true });
                pipe.dispatch(ctx, lab.cmds.cmd, &push, .{ 1, 1, 1 });
            }
            try lab.cmds.submitAndWait(ctx);
            one[r] = @as(f64, @floatFromInt(clock.nowNs() - t0)) / @as(f64, @floatFromInt(k));

            t0 = clock.nowNs();
            for (0..k) |_| {
                try lab.cmds.begin(ctx);
                pipe.dispatch(ctx, lab.cmds.cmd, &push, .{ 1, 1, 1 });
                try lab.cmds.submitAndWait(ctx);
            }
            each[r] = @as(f64, @floatFromInt(clock.nowNs() - t0)) / @as(f64, @floatFromInt(k));
        }
        const kx: f64 = @floatFromInt(k);
        var ns = report.median(one);
        try report.write(out, .{ .device = "link", .probe = "batch", .variant = "one-submit", .x = kx, .value = ns / 1e3, .unit = "us", .ns = ns });
        ns = report.median(each);
        try report.write(out, .{ .device = "link", .probe = "batch", .variant = "submit-each", .x = kx, .value = ns / 1e3, .unit = "us", .ns = ns });
    }
}

/// Where one small round trip's time goes, on the CPU's clock (GPU timestamps placed on it with
/// a calibration taken right after the trip):
///   record  CPU: recording the command buffer
///   submit  CPU: inside vkQueueSubmit2
///   launch  submit called -> the GPU starts the buffer (the submit call overlaps this)
///   run     the GPU runs it (one tiny dispatch between two timestamps)
///   wake    the GPU finished -> the waiting CPU thread runs again
/// record + launch + run + wake = total exactly; `submit` is part of `launch`.
pub const Phases = struct {
    record: f64,
    submit: f64,
    launch: f64,
    run: f64,
    wake: f64,
    total: f64,
    deviation: f64,

    pub const names = [_][]const u8{ "record", "submit", "launch", "run", "wake", "total", "deviation" };
};

fn trip(lab: *Lab, pipe: *const gpu.ComputePipeline, push: *const gpu_probes.FmaPush, mode: gpu.Commands.Wait) !Phases {
    const ctx = lab.ctx;
    const c0 = clock.nowNs();
    try lab.cmds.begin(ctx);
    lab.stamps.reset(ctx, lab.cmds.cmd);
    lab.stamps.write(ctx, lab.cmds.cmd, 0);
    pipe.dispatch(ctx, lab.cmds.cmd, push, .{ 1, 1, 1 });
    lab.stamps.write(ctx, lab.cmds.cmd, 1);
    try ctx.vkd.endCommandBuffer(lab.cmds.cmd);
    const c1 = clock.nowNs();
    try lab.cmds.submit(ctx);
    const c2 = clock.nowNs();
    try lab.cmds.wait(ctx, mode);
    const c3 = clock.nowNs();
    const cal = try gpu.Calibration.take(ctx);
    const g0 = cal.cpuNsOf(try lab.stamps.tick(ctx, 0), c0);
    const g1 = cal.cpuNsOf(try lab.stamps.tick(ctx, 1), c0);
    const rec: f64 = @floatFromInt(c1 - c0);
    const sub: f64 = @floatFromInt(c2 - c1);
    const total: f64 = @floatFromInt(c3 - c0);
    return .{ .record = rec, .submit = sub, .launch = g0 - rec, .run = g1 - g0, .wake = total - g1, .total = total, .deviation = @floatFromInt(cal.deviation_ns) };
}

/// Median of each phase over `ps` (medians of the parts need not sum to the median total).
fn medianPhases(ps: []Phases, scratch: []f64) Phases {
    var m: Phases = undefined;
    inline for (Phases.names) |f| {
        for (ps, scratch) |p, *s| s.* = @field(p, f);
        @field(m, f) = report.median(scratch);
    }
    return m;
}

fn writePhases(out: *std.Io.Writer, probe: []const u8, prefix: []const u8, x: f64, m: Phases) !void {
    var buf: [48]u8 = undefined;
    inline for (Phases.names) |f| {
        const ns = @field(m, f);
        try report.write(out, .{ .device = "link", .probe = probe, .variant = try std.mem.print(&buf, "{s}-{s}", .{ prefix, f }), .x = x, .value = ns / 1e3, .unit = "us", .ns = ns });
    }
}

fn phasesReady(gpa: std.mem.Allocator, lab: *Lab, probe: []const u8) !bool {
    if (try gpu.Calibration.available(lab.ctx, gpa)) return true;
    report.note("link {s}: skipped, the driver cannot calibrate GPU timestamps against the CPU clock\n", .{probe});
    return false;
}

/// The round trip split into phases, for a blocking and a polling wait.
pub fn probePhases(gpa: std.mem.Allocator, lab: *Lab, out: *std.Io.Writer, quick: bool) !void {
    if (!try phasesReady(gpa, lab, "phases")) return;
    const ctx = lab.ctx;
    const rounds: usize = if (quick) 5 else 200;
    var pipe = try lab.pipeline(gpu_probes.fma32Code(), &.{lab.out.handle}, @sizeOf(gpu_probes.FmaPush), .{ .subgroup_size = 64, .spec = &.{ 1, 1 } });
    defer pipe.deinit(ctx);
    const push: gpu_probes.FmaPush = .{ .iters = 1, .lanes = 64, .a = 1, .b = 1 };
    const ps = try gpa.alloc(Phases, rounds);
    defer gpa.free(ps);
    const scratch = try gpa.alloc(f64, rounds);
    defer gpa.free(scratch);
    inline for (.{ .block, .spin }) |mode| {
        for (ps) |*p| p.* = try trip(lab, &pipe, &push, mode);
        try writePhases(out, "phases", @tagName(mode), 0, medianPhases(ps, scratch));
    }
}

/// Does the GPU get slower to answer after sitting idle (clocks or power state)? One round trip
/// after `gap` ms of idle ("cold"), and one after the same idle plus a throwaway trip ("warm").
pub fn probeIdle(gpa: std.mem.Allocator, lab: *Lab, out: *std.Io.Writer, quick: bool) !void {
    if (!try phasesReady(gpa, lab, "idle")) return;
    const ctx = lab.ctx;
    var pipe = try lab.pipeline(gpu_probes.fma32Code(), &.{lab.out.handle}, @sizeOf(gpu_probes.FmaPush), .{ .subgroup_size = 64, .spec = &.{ 1, 1 } });
    defer pipe.deinit(ctx);
    const push: gpu_probes.FmaPush = .{ .iters = 1, .lanes = 64, .a = 1, .b = 1 };
    const Gap = struct { ms: u32, rounds: usize };
    const gaps: []const Gap = if (quick)
        &.{ .{ .ms = 0, .rounds = 2 }, .{ .ms = 1, .rounds = 2 } }
    else
        &.{ .{ .ms = 0, .rounds = 20 }, .{ .ms = 1, .rounds = 20 }, .{ .ms = 10, .rounds = 20 }, .{ .ms = 100, .rounds = 20 }, .{ .ms = 1000, .rounds = 5 } };
    var cold: [20]Phases = undefined;
    var warm: [20]Phases = undefined;
    var scratch: [20]f64 = undefined;
    for (gaps) |g| {
        for (0..g.rounds) |r| {
            if (g.ms > 0) host.sleepMs(g.ms);
            cold[r] = try trip(lab, &pipe, &push, .block);
            if (g.ms > 0) host.sleepMs(g.ms);
            _ = try trip(lab, &pipe, &push, .block);
            warm[r] = try trip(lab, &pipe, &push, .block);
        }
        const x: f64 = @floatFromInt(g.ms);
        try writePhases(out, "idle", "cold", x, medianPhases(cold[0..g.rounds], scratch[0..g.rounds]));
        try writePhases(out, "idle", "warm", x, medianPhases(warm[0..g.rounds], scratch[0..g.rounds]));
    }
}

pub const all = [_][]const u8{ "upload", "download", "roundtrip", "wait", "batch", "phases", "idle" };

pub fn run(gpa: std.mem.Allocator, lab: *Lab, out: *std.Io.Writer, name: []const u8, quick: bool) !bool {
    if (std.mem.eql(u8, name, "upload")) try probeUpload(gpa, lab, out, quick) //
    else if (std.mem.eql(u8, name, "download")) try probeDownload(gpa, lab, out, quick) //
    else if (std.mem.eql(u8, name, "roundtrip")) try probeRoundtrip(gpa, lab, out, quick) //
    else if (std.mem.eql(u8, name, "wait")) try probeWait(gpa, lab, out, quick) //
    else if (std.mem.eql(u8, name, "batch")) try probeBatch(gpa, lab, out, quick) //
    else if (std.mem.eql(u8, name, "phases")) try probePhases(gpa, lab, out, quick) //
    else if (std.mem.eql(u8, name, "idle")) try probeIdle(gpa, lab, out, quick) //
    else return false;
    return true;
}

test "link calibration places GPU ticks on the CPU clock" {
    // 10 ns per tick; calibration says GPU tick 1000 happened at CPU 5_000_000 ns.
    const cal: gpu.Calibration = .{ .gpu_ticks = 1000, .cpu_ns = 5_000_000, .deviation_ns = 0, .period_ns = 10 };
    try std.testing.expectEqual(@as(f64, 0), cal.cpuNsOf(1000, 5_000_000));
    try std.testing.expectEqual(@as(f64, 500), cal.cpuNsOf(1050, 5_000_000));
    // A tick before the calibration lands before it, and a base after the calibration gives a
    // negative offset: both are signed, not wrapped.
    try std.testing.expectEqual(@as(f64, -200), cal.cpuNsOf(980, 5_000_000));
    try std.testing.expectEqual(@as(f64, -1000), cal.cpuNsOf(1000, 5_001_000));
}

test "link phases add up and sit on the CPU's clock" {
    report.progress = false;
    var feats: gpu_probes.Features = .{};
    var ctx = try gpu.Context.init(std.testing.allocator, feats.options(true));
    defer ctx.deinit();
    if (!try gpu.Calibration.available(&ctx, std.testing.allocator)) return error.SkipZigTest;
    var lab = try Lab.init(&ctx, true);
    defer lab.deinit();
    var pipe = try lab.pipeline(gpu_probes.fma32Code(), &.{lab.out.handle}, @sizeOf(gpu_probes.FmaPush), .{ .subgroup_size = 64, .spec = &.{ 1, 1 } });
    defer pipe.deinit(&ctx);
    const push: gpu_probes.FmaPush = .{ .iters = 1, .lanes = 64, .a = 1, .b = 1 };
    inline for (.{ .block, .spin }) |mode| {
        for (0..5) |_| {
            const p = try trip(&lab, &pipe, &push, mode);
            try std.testing.expectApproxEqAbs(p.total, p.record + p.launch + p.run + p.wake, 1e-3);
            try std.testing.expect(p.run >= 0);
            // The GPU starts after the submit is called and finishes before the CPU wakes, up to
            // the calibration's own error and the CPU clock's tick (QPC: 100 ns). A wrong clock
            // domain or unit puts these off by seconds.
            const slack = p.deviation + 2000;
            try std.testing.expect(p.launch > -slack and p.launch < p.total + slack);
            try std.testing.expect(p.wake > -slack and p.wake < p.total + slack);
        }
    }
    try std.testing.expectEqual(@as(u32, 0), gpu.validation_messages.load(.monotonic));
}

test "link every probe runs (small sizes)" {
    report.progress = false;
    var feats: gpu_probes.Features = .{};
    var ctx = try gpu.Context.init(std.testing.allocator, feats.options(true));
    defer ctx.deinit();
    var lab = try Lab.init(&ctx, true);
    defer lab.deinit();
    var sink: std.Io.Writer.Discarding = .init(&.{});
    for (all) |name| try std.testing.expect(try run(std.testing.allocator, &lab, &sink.writer, name, true));
    try std.testing.expectEqual(@as(u32, 0), gpu.validation_messages.load(.monotonic));
}
