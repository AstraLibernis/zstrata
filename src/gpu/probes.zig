// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! The GPU side of the map (first built in an earlier Vulkan lab project). One probe per tier of the card, then
//! the tiers combined:
//!
//!   alu       fp32 / packed fp16 multiply-adds by values kept live per lane (registers), wave32 vs wave64
//!   lanes     the same work with only the first L lanes of each subgroup busy
//!   occupancy registers per lane x shared memory per workgroup, both of which cap how many
//!             subgroups a compute unit can hold at once
//!   lds       shared-memory bandwidth by access stride (bank conflicts)
//!   stream    read bandwidth by working-set size (caches -> VRAM)
//!   chase     load latency by working-set size
//!   coop      matrix-unit peak by number type (fp16, bf16, fp8, int8)
//!   mlp       combined: networks of width W with weights in registers / shared memory / the
//!             buffer, one or several networks per subgroup
//!
//! Every row is the median of 5 timed runs (GPU timestamps around one dispatch), each sized
//! to ~40 ms after calibration. Compile statistics (registers, spills, scratch, shared memory,
//! subgroups per SIMD) come from the driver through VK_KHR_pipeline_executable_properties.

const std = @import("std");
const builtin = @import("builtin");
const gpu = @import("../gpu.zig");
const report = @import("../report.zig");
const vk = gpu.vk;

const spv = struct {
    const fma32 align(@alignOf(u32)) = @embedFile("fma32.spv").*;
    const fma16 align(@alignOf(u32)) = @embedFile("fma16.spv").*;
    const lds_bw align(@alignOf(u32)) = @embedFile("lds_bw.spv").*;
    const stream align(@alignOf(u32)) = @embedFile("stream.spv").*;
    const chase align(@alignOf(u32)) = @embedFile("chase.spv").*;
    const coop_f16 align(@alignOf(u32)) = @embedFile("coop_f16.spv").*;
    const coop_bf16 align(@alignOf(u32)) = @embedFile("coop_bf16.spv").*;
    const coop_f8 align(@alignOf(u32)) = @embedFile("coop_f8.spv").*;
    const coop_i8 align(@alignOf(u32)) = @embedFile("coop_i8.spv").*;
    const mlp_reg align(@alignOf(u32)) = @embedFile("mlp_reg.spv").*;
    const mlp_lds align(@alignOf(u32)) = @embedFile("mlp_lds.spv").*;
    const mlp_lds4 align(@alignOf(u32)) = @embedFile("mlp_lds4.spv").*;
    const mlp_mem align(@alignOf(u32)) = @embedFile("mlp_mem.spv").*;
    const mlp_mem4 align(@alignOf(u32)) = @embedFile("mlp_mem4.spv").*;
};

/// Feature structs on top of the defaults: compile statistics, matrix units for every
/// number type, fp16/int8 arithmetic in shaders.
pub const Features = struct {
    exec: vk.PhysicalDevicePipelineExecutablePropertiesFeaturesKHR = .{ .pipeline_executable_info = .true },
    coop: vk.PhysicalDeviceCooperativeMatrixFeaturesKHR = .{ .cooperative_matrix = .true },
    bf16: vk.PhysicalDeviceShaderBfloat16FeaturesKHR = .{ .shader_b_float_16_type = .true, .shader_b_float_16_cooperative_matrix = .true },
    f8: vk.PhysicalDeviceShaderFloat8FeaturesEXT = .{ .shader_float_8 = .true, .shader_float_8_cooperative_matrix = .true },
    v12: vk.PhysicalDeviceVulkan12Features = .{ .shader_float_16 = .true, .shader_int_8 = .true, .vulkan_memory_model = .true },

    /// Chain the structs together; `self` must not move afterwards.
    pub fn options(self: *Features, validation: bool) gpu.Options {
        self.exec.p_next = &self.coop;
        self.coop.p_next = &self.bf16;
        self.bf16.p_next = &self.f8;
        self.f8.p_next = &self.v12;
        return .{
            .validation = validation,
            .device_extensions = &.{
                vk.device_extensions.khr_pipeline_executable_properties.name,
                vk.device_extensions.khr_cooperative_matrix.name,
                vk.device_extensions.khr_shader_bfloat_16.name,
                vk.device_extensions.ext_shader_float_8.name,
            },
            // The link's phase split needs it; without it those probes skip themselves.
            .optional_extensions = &.{gpu.Calibration.extension},
            .device_features = &self.exec,
        };
    }
};

/// The driver's compile statistics for one pipeline (see report.Stats).
/// Compile statistics under RADV's names, else AMD's Windows driver's (`gpu --list-stats` prints
/// what a driver reports). AMD's has no spill, occupancy, VOPD or SMEM counts: those stay empty.
pub fn statsOf(ctx: *const gpu.Context, pipe: *const gpu.ComputePipeline) !report.Stats {
    return .{
        .vgprs = try stat(ctx, pipe, &.{ "VGPRs", "numUsedVgprs" }),
        .sgprs = try stat(ctx, pipe, &.{ "SGPRs", "numUsedSgprs" }),
        .spilled_vgprs = try stat(ctx, pipe, &.{"Spilled VGPRs"}),
        .scratch = try stat(ctx, pipe, &.{ "Scratch size", "scratchMemUsageInBytes" }),
        // Both are what is allocated per workgroup (RADV rounds up the same way).
        .lds = try stat(ctx, pipe, &.{ "LDS size", "ldsSizePerLocalWorkGroup" }),
        .waves = try stat(ctx, pipe, &.{"Subgroups per SIMD"}),
        .vopd = try stat(ctx, pipe, &.{"VOPD"}),
        .smem = try stat(ctx, pipe, &.{"SMEM"}),
    };
}

/// The first of `names` the driver reports, or null.
fn stat(ctx: *const gpu.Context, pipe: *const gpu.ComputePipeline, names: []const []const u8) !?u64 {
    for (names) |n| if (try pipe.statistic(ctx, n)) |v| return v;
    return null;
}

/// Shared timing rig: one command buffer, two timestamps, an output buffer big enough for
/// every probe, and a 1 GiB buffer for the memory probes and the network weights.
pub const Lab = struct {
    ctx: *const gpu.Context,
    cmds: gpu.Commands,
    stamps: gpu.Timestamps,
    out: gpu.Buffer,
    big: gpu.Buffer,
    quick: bool,

    const out_bytes = 64 << 20;
    const big_bytes = 1 << 30;
    /// Each timed run aims at this much GPU time; one submit far above ~1.5 s resets the GPU.
    const target_ns = 40e6;
    const reps = 5;

    pub fn init(ctx: *const gpu.Context, quick: bool) !Lab {
        var cmds = try gpu.Commands.init(ctx);
        errdefer cmds.deinit(ctx);
        var stamps = try gpu.Timestamps.init(ctx, 2);
        errdefer stamps.deinit(ctx);
        const usage: vk.BufferUsageFlags = .{ .storage_buffer = true, .transfer_dst = true, .transfer_src = true };
        var out = try gpu.Buffer.init(ctx, out_bytes, usage, .{ .device_local = true }, .{});
        errdefer out.deinit(ctx);
        var big = try gpu.Buffer.init(ctx, if (quick) 64 << 20 else big_bytes, usage, .{ .device_local = true }, .{});
        errdefer big.deinit(ctx);
        return .{ .ctx = ctx, .cmds = cmds, .stamps = stamps, .out = out, .big = big, .quick = quick };
    }

    pub fn deinit(self: *Lab) void {
        self.big.deinit(self.ctx);
        self.out.deinit(self.ctx);
        self.stamps.deinit(self.ctx);
        self.cmds.deinit(self.ctx);
        self.* = undefined;
    }

    pub fn bigBytes(self: *const Lab) u64 {
        return if (self.quick) 64 << 20 else big_bytes;
    }

    /// GPU time of one dispatch, in its own submit.
    /// The most shared memory one workgroup may declare, as the driver reports it.
    pub fn sharedLimit(self: *const Lab) u64 {
        return self.ctx.props.limits.max_compute_shared_memory_size;
    }

    pub fn once(self: *Lab, pipe: *const gpu.ComputePipeline, push: *const anyopaque, groups: u32) !f64 {
        const ctx = self.ctx;
        const cmd = self.cmds.cmd;
        try self.cmds.begin(ctx);
        self.stamps.reset(ctx, cmd);
        self.stamps.write(ctx, cmd, 0);
        pipe.dispatch(ctx, cmd, push, .{ groups, 1, 1 });
        self.stamps.write(ctx, cmd, 1);
        try self.cmds.submitAndWait(ctx);
        return self.stamps.elapsedNs(ctx, 0, 1);
    }

    /// Grow `push.iters` (from the value it holds) until one run takes ~target_ns, then return
    /// the median of `reps` runs. `push.iters` keeps the final value, for the work count.
    pub fn measure(self: *Lab, pipe: *const gpu.ComputePipeline, push: anytype, groups: u32) !f64 {
        if (self.quick) return self.once(pipe, push, groups);
        var t = try self.once(pipe, push, groups);
        while (t < target_ns / 8) {
            push.iters *= if (t < target_ns / 64) 8 else 2;
            t = try self.once(pipe, push, groups);
        }
        const scaled = @as(f64, @floatFromInt(push.iters)) * target_ns / t;
        push.iters = @intFromFloat(@max(1, @min(scaled, 4e9)));
        _ = try self.once(pipe, push, groups); // the clocks settle on the new size
        var xs: [reps]f64 = undefined;
        for (&xs) |*x| x.* = try self.once(pipe, push, groups);
        std.mem.sort(f64, &xs, {}, std.sort.asc(f64));
        return xs[reps / 2];
    }

    pub fn pipeline(self: *const Lab, code: []align(4) const u8, buffers: []const vk.Buffer, push_size: u32, extra: gpu.ComputePipeline.Extra) !gpu.ComputePipeline {
        const kinds: [8]vk.DescriptorType = @splat(.storage_buffer);
        var e = extra;
        e.capture_statistics = true;
        var p = try gpu.ComputePipeline.initExtra(self.ctx, code, kinds[0..buffers.len], push_size, 1, e);
        p.bindBuffers(self.ctx, buffers);
        return p;
    }
};

// ---------------------------------------------------------------- ALU, lanes, occupancy

pub const FmaPush = extern struct { iters: u32, lanes: u32, a: f32, b: f32 };
const fma_groups = 4096; // workgroups of 256: 1 M lanes, far more than the card holds at once

fn fmaRun(lab: *Lab, out: *std.Io.Writer, probe: []const u8, variant: []const u8, half: bool, wave: u32, k: u32, shared_floats: u32, lanes: u32, x: f64, y: f64) !void {
    const code: []align(4) const u8 = if (half) &spv.fma16 else &spv.fma32;
    var pipe = try lab.pipeline(code, &.{lab.out.handle}, @sizeOf(FmaPush), .{ .subgroup_size = wave, .spec = &.{ k, shared_floats } });
    defer pipe.deinit(lab.ctx);
    var push: FmaPush = .{ .iters = 16, .lanes = lanes, .a = 0.999, .b = 0.001 };
    const ns = try lab.measure(&pipe, &push, fma_groups);
    const busy_lanes = @as(f64, fma_groups * 256 / wave) * @as(f64, @floatFromInt(lanes));
    const flops = busy_lanes * @as(f64, @floatFromInt(k)) * @as(f64, @floatFromInt(push.iters)) * @as(f64, if (half) 4.0 else 2.0);
    try report.write(out, .{ .device = "gpu", .probe = probe, .variant = variant, .x = x, .y = y, .value = flops / ns / 1e3, .unit = "TFLOP/s", .ns = ns, .stats = try statsOf(lab.ctx, &pipe) });
}

fn probeAlu(lab: *Lab, out: *std.Io.Writer) !void {
    const ks: []const u32 = if (lab.quick) &.{ 4, 64 } else &.{ 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 512 };
    for ([_]bool{ false, true }) |half| for ([_]u32{ 64, 32 }) |wave| {
        const name = if (half) (if (wave == 64) "fp16x2-wave64" else "fp16x2-wave32") else (if (wave == 64) "fp32-wave64" else "fp32-wave32");
        for (ks) |k| try fmaRun(lab, out, "alu", name, half, wave, k, 1, wave, @floatFromInt(k), 0);
    };
}

fn probeLanes(lab: *Lab, out: *std.Io.Writer) !void {
    for ([_]u32{ 64, 32 }) |wave| {
        const name = if (wave == 64) "fp32-wave64" else "fp32-wave32";
        const ls: []const u32 = if (lab.quick) &.{ 1, wave } else if (wave == 64)
            &.{ 1, 2, 4, 8, 16, 24, 32, 33, 40, 48, 56, 63, 64 }
        else
            &.{ 1, 2, 4, 8, 12, 16, 20, 24, 28, 31, 32 };
        for (ls) |l| try fmaRun(lab, out, "lanes", name, false, wave, 32, 1, l, @floatFromInt(l), 0);
    }
}

fn probeOccupancy(lab: *Lab, out: *std.Io.Writer) !void {
    const ks: []const u32 = if (lab.quick) &.{4} else &.{ 2, 4, 8, 16, 32, 64, 128, 256 };
    const kib: []const u32 = if (lab.quick) &.{ 0, 64 } else &.{ 0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64 };
    for (ks) |k| for (kib) |s| {
        // Past the device's shared-memory limit the pipeline is invalid (the spec leaves it
        // undefined): AMD's Windows driver reports 32 KiB where RADV reports 64 KiB.
        if (@as(u64, s) * 1024 > lab.sharedLimit()) {
            if (k == ks[0]) report.note("occupancy {d} KiB shared: above the device limit of {d} KiB, skipped\n", .{ s, lab.sharedLimit() / 1024 });
            continue;
        }
        try fmaRun(lab, out, "occupancy", "fp32-wave64", false, 64, k, @max(1, s * 256), 64, @floatFromInt(k), @floatFromInt(s));
    };
}

// ---------------------------------------------------------------- shared memory

const LdsPush = extern struct { iters: u32, stride: u32 };

fn probeLds(lab: *Lab, out: *std.Io.Writer) !void {
    var pipe = try lab.pipeline(&spv.lds_bw, &.{lab.out.handle}, @sizeOf(LdsPush), .{ .subgroup_size = 64 });
    defer pipe.deinit(lab.ctx);
    const stats = try statsOf(lab.ctx, &pipe);
    const strides: []const u32 = if (lab.quick) &.{ 1, 32 } else &.{ 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 24, 31, 32, 33, 48, 63, 64, 65, 128 };
    for (strides) |s| {
        var push: LdsPush = .{ .iters = 4, .stride = s };
        const groups = 2048;
        const ns = try lab.measure(&pipe, &push, groups);
        const bytes = @as(f64, groups * 256) * @as(f64, @floatFromInt(push.iters)) * 16 * 4;
        try report.write(out, .{ .device = "gpu", .probe = "lds", .variant = "read-fp32", .x = @floatFromInt(s), .value = bytes / ns, .unit = "GB/s", .ns = ns, .stats = stats });
    }
}

// ---------------------------------------------------------------- caches and VRAM

/// Working-set sizes from 4 KiB to the big buffer, in steps of x sqrt(2).
fn sizesList(buf: []u64, max: u64) []u64 {
    var n: usize = 0;
    var s: u64 = 4 << 10;
    while (s <= max and n + 1 < buf.len) : (s *= 2) {
        buf[n] = s;
        n += 1;
        const mid = s + s / 2; // 1.5x; masks need a power of two, so stream skips these
        if (mid <= max) {
            buf[n] = mid;
            n += 1;
        }
    }
    return buf[0..n];
}

/// Workgroups per stream dispatch. Every one must be on the GPU at once (shaders/stream.comp):
/// on the RX 9070 XT 512 workgroups of 256 lanes fill every wave slot, and 1024 ran in two
/// batches, which doubled every cache size. Two grids, because the L2 reads faster with fewer
/// workgroups in flight (docs/measurements.md).
const stream_groups = [_]u32{ 256, 512 };

fn probeStream(lab: *Lab, out: *std.Io.Writer) !void {
    // Random contents: RDNA4 compresses memory, and zeros read up to 23% faster from the caches.
    try fillRandom(lab, lab.big.handle, lab.bigBytes(), 0x5eed);
    var pipe = try lab.pipeline(&spv.stream, &.{ lab.big.handle, lab.out.handle }, @sizeOf(StreamPush), .{ .subgroup_size = 64 });
    defer pipe.deinit(lab.ctx);
    const stats = try statsOf(lab.ctx, &pipe);
    var size_buf: [64]u64 = undefined;
    for (stream_groups) |groups| {
        for (sizesList(&size_buf, lab.bigBytes())) |s| {
            // Whole 4 KiB blocks only. Below groups * 4 KiB several workgroups share a block,
            // which still measures the nearest cache.
            if (s % 4096 != 0) continue;
            var push: StreamPush = .{ .nblocks = @intCast(s / 4096), .iters = 1 };
            const ns = try lab.measure(&pipe, &push, groups);
            const bytes = @as(f64, @floatFromInt(groups * 256)) * @as(f64, @floatFromInt(push.iters)) * 16;
            try report.write(out, .{ .device = "gpu", .probe = "stream", .variant = "sweep", .x = @floatFromInt(s), .y = @floatFromInt(groups), .value = bytes / ns, .unit = "GB/s", .ns = ns, .stats = stats });
        }
    }
}

const StreamPush = extern struct { nblocks: u32, iters: u32 };

/// Fill the first `bytes` of `dst` with seeded random bytes, through a staging buffer of at
/// most 64 MiB.
fn fillRandom(lab: *Lab, dst: vk.Buffer, bytes: u64, seed: u64) !void {
    const chunk: u64 = @min(bytes, 64 << 20);
    var staging = try gpu.Buffer.init(lab.ctx, chunk, .{ .transfer_src = true }, .{ .host_visible = true, .host_coherent = true }, .{});
    defer staging.deinit(lab.ctx);
    var rng = std.Random.DefaultPrng.init(seed);
    var off: u64 = 0;
    while (off < bytes) : (off += chunk) {
        const n = @min(chunk, bytes - off);
        rng.random().bytes(staging.mapped.?[0..n]);
        try lab.cmds.begin(lab.ctx);
        lab.ctx.vkd.cmdCopyBuffer(lab.cmds.cmd, staging.handle, dst, &.{.{ .src_offset = 0, .dst_offset = off, .size = n }});
        try lab.cmds.submitAndWait(lab.ctx);
    }
}

fn probeChase(gpa: std.mem.Allocator, lab: *Lab, out: *std.Io.Writer) !void {
    var pipes: [2]gpu.ComputePipeline = undefined;
    var made: usize = 0;
    defer for (pipes[0..made]) |*p| p.deinit(lab.ctx);
    for (&pipes, 0..) |*p, vector| {
        p.* = try lab.pipeline(&spv.chase, &.{ lab.big.handle, lab.out.handle }, @sizeOf(ChasePush), .{ .subgroup_size = 64, .spec = &.{@intCast(vector)} });
        made += 1;
    }
    const stats = [2]report.Stats{ try statsOf(lab.ctx, &pipes[0]), try statsOf(lab.ctx, &pipes[1]) };
    const max: u64 = if (lab.quick) 1 << 20 else 512 << 20;
    var size_buf: [64]u64 = undefined;
    var rng = std.Random.DefaultPrng.init(10);
    for (sizesList(&size_buf, max)) |s| {
        const ring = try gpa.alloc(u32, s / 4);
        defer gpa.free(ring);
        buildRing(ring, rng.random());
        try gpu.uploadOnce(lab.ctx, &lab.cmds, lab.big.handle, std.mem.sliceAsBytes(ring));
        for (&pipes, stats, [_][]const u8{ "scalar", "vector" }) |*pipe, st, name| {
            var push: ChasePush = .{ .iters = 256 };
            const ns = try measureChase(lab, pipe, &push, rng.random(), @intCast(s / 128));
            try report.write(out, .{ .device = "gpu", .probe = "chase", .variant = name, .x = @floatFromInt(s), .value = ns / @as(f64, @floatFromInt(push.iters)), .unit = "ns", .ns = ns, .stats = st });
        }
    }
}

/// Like Lab.measure, but every run starts at a fresh random line of the ring.
fn measureChase(lab: *Lab, pipe: *const gpu.ComputePipeline, push: *ChasePush, rand: std.Random, lines: u32) !f64 {
    const Fresh = struct {
        fn run(l: *Lab, p: *const gpu.ComputePipeline, ps: *ChasePush, r: std.Random, n: u32) !f64 {
            ps.start = r.uintLessThan(u32, n) * 32;
            return l.once(p, ps, 1);
        }
    };
    if (lab.quick) return Fresh.run(lab, pipe, push, rand, lines);
    var t = try Fresh.run(lab, pipe, push, rand, lines);
    while (t < Lab.target_ns / 8) {
        push.iters *= if (t < Lab.target_ns / 64) 8 else 2;
        t = try Fresh.run(lab, pipe, push, rand, lines);
    }
    push.iters = @intFromFloat(@max(1, @min(@as(f64, @floatFromInt(push.iters)) * Lab.target_ns / t, 4e9)));
    var xs: [Lab.reps]f64 = undefined;
    for (&xs) |*x| x.* = try Fresh.run(lab, pipe, push, rand, lines);
    std.mem.sort(f64, &xs, {}, std.sort.asc(f64));
    return xs[Lab.reps / 2];
}

const ChasePush = extern struct { iters: u32, mul: u32 = 0, start: u32 = 0 };

/// One random cycle through every 128-byte line of `ring` (Sattolo's shuffle): ring[p] holds
/// the index of the next line's first u32.
fn buildRing(ring: []u32, rand: std.Random) void {
    const line = 32; // u32s per 128-byte cache line
    const n = ring.len / line;
    @memset(ring, 0);
    const order = std.heap.page_allocator.alloc(u32, n) catch @panic("OOM");
    defer std.heap.page_allocator.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    var i = n;
    while (i > 1) {
        i -= 1;
        const j = rand.uintLessThan(usize, i);
        std.mem.swap(u32, &order[i], &order[j]);
    }
    for (0..n) |k| ring[order[k] * line] = order[(k + 1) % n] * line;
}

// ---------------------------------------------------------------- matrix units

const CoopPush = extern struct { iters: u32 };

fn probeCoop(gpa: std.mem.Allocator, lab: *Lab, out: *std.Io.Writer) !void {
    const ctx = lab.ctx;
    var count: u32 = 0;
    _ = try ctx.vki.getPhysicalDeviceCooperativeMatrixPropertiesKHR(ctx.physical, &count, null);
    const props = try gpa.alloc(vk.CooperativeMatrixPropertiesKHR, count);
    defer gpa.free(props);
    for (props) |*p| p.* = std.mem.zeroInit(vk.CooperativeMatrixPropertiesKHR, .{ .s_type = .cooperative_matrix_properties_khr });
    _ = try ctx.vki.getPhysicalDeviceCooperativeMatrixPropertiesKHR(ctx.physical, &count, props.ptr);
    for (props) |p| report.note("coopmat {d}x{d}x{d} A={t} C={t} result={t} scope={t}\n", .{ p.m_size, p.n_size, p.k_size, p.a_type, p.c_type, p.result_type, p.scope });

    const Kind = struct { name: []const u8, code: []align(4) const u8, a: vk.ComponentTypeKHR, c: vk.ComponentTypeKHR, unit: []const u8 };
    const kinds = [_]Kind{
        .{ .name = "fp16-fp32acc", .code = &spv.coop_f16, .a = .float16_khr, .c = .float32_khr, .unit = "TFLOP/s" },
        .{ .name = "bf16-fp32acc", .code = &spv.coop_bf16, .a = .bfloat16_khr, .c = .float32_khr, .unit = "TFLOP/s" },
        .{ .name = "fp8e4m3-fp32acc", .code = &spv.coop_f8, .a = .float8_e4m3_ext, .c = .float32_khr, .unit = "TFLOP/s" },
        .{ .name = "int8-int32acc", .code = &spv.coop_i8, .a = .sint8_khr, .c = .sint32_khr, .unit = "TOP/s" },
    };
    for (kinds, 0..) |kd, idx| {
        const supported = for (props) |p| {
            if (p.m_size == 16 and p.n_size == 16 and p.k_size == 16 and p.a_type == kd.a and p.c_type == kd.c) break true;
        } else false;
        if (!supported) {
            report.note("coop {s}: not offered at 16x16x16, skipped\n", .{kd.name});
            continue;
        }
        var pipe = try lab.pipeline(kd.code, &.{lab.out.handle}, @sizeOf(CoopPush), .{ .subgroup_size = 64 });
        defer pipe.deinit(ctx);
        var push: CoopPush = .{ .iters = 16 };
        const groups = 2048; // 4 subgroups each
        const ns = try lab.measure(&pipe, &push, groups);
        const ops = @as(f64, groups * 4) * @as(f64, @floatFromInt(push.iters)) * 8 * 2 * 16 * 16 * 16;
        try report.write(out, .{ .device = "gpu", .probe = "coop", .variant = kd.name, .x = @floatFromInt(idx), .value = ops / ns / 1e3, .unit = kd.unit, .ns = ns, .stats = try statsOf(ctx, &pipe) });
    }
}

// ---------------------------------------------------------------- combined: networks

const MlpPush = extern struct { iters: u32, layers: u32 };
const mlp_layers = 4;
const mlp_groups = 2048; // one 64-lane subgroup each
const weight_scale = 1.0 / 16.0;

pub const Place = enum {
    reg,
    lds,
    /// `lds` read 16 bytes at a time (`LDS_VEC4` in mlp.glsl): the same arithmetic in the same
    /// order, so results are bit-equal; only the load width differs.
    lds4,
    mem,
    /// `mem` with the weights read from the buffer as vec4 (`MEM_VEC4`); bit-equal results.
    mem4,

    fn code(self: Place) []align(4) const u8 {
        return switch (self) {
            .reg => &spv.mlp_reg,
            .lds => &spv.mlp_lds,
            .lds4 => &spv.mlp_lds4,
            .mem => &spv.mlp_mem,
            .mem4 => &spv.mlp_mem4,
        };
    }
};

/// Shared memory one workgroup of the network probe needs.
fn mlpShared(place: Place, w: u32, nps: u32) u64 {
    const acts = @as(u64, nps) * w * 4;
    return if (place == .lds or place == .lds4) acts + @as(u64, nps) * w * w * 4 else acts;
}

fn probeMlp(gpa: std.mem.Allocator, lab: *Lab, out: *std.Io.Writer) !void {
    const ctx = lab.ctx;
    // Weights for the largest case fill the front of the big buffer once; smaller cases use a
    // prefix of it. Uniform in [-1, 1) / 16.
    const max_w: u32 = if (lab.quick) 32 else 256;
    const floats: usize = @min(@as(usize, mlp_groups) * @max(max_w, 64) * max_w, lab.bigBytes() / 4);
    const weights = try gpa.alloc(f32, floats);
    defer gpa.free(weights);
    var rng = std.Random.DefaultPrng.init(42);
    for (weights) |*v| v.* = (rng.random().float(f32) * 2 - 1) * weight_scale;
    try gpu.uploadOnce(ctx, &lab.cmds, lab.big.handle, std.mem.sliceAsBytes(weights));

    const widths: []const u32 = if (lab.quick) &.{ 8, 32 } else &.{ 8, 16, 24, 32, 48, 64, 96, 128, 192, 256 };
    for (widths) |w| for ([_]Place{ .reg, .lds, .lds4, .mem, .mem4 }) |place| {
        const packed_nps: u32 = if (w < 64) 64 / w else 1;
        for ([_]u32{ 1, packed_nps }, 0..) |nps, pass| {
            if (pass == 1 and nps == 1) continue;
            const variant = try std.fmt.allocPrint(gpa, "{t}-{s}", .{ place, if (pass == 0) "one" else "packed" });
            defer gpa.free(variant);
            const nets = mlp_groups * nps;
            if (mlpShared(place, w, nps) > lab.sharedLimit() or @as(u64, nets) * w * w > floats) {
                report.note("mlp {s} W={d}: does not fit, skipped\n", .{ variant, w });
                continue;
            }
            var pipe = try lab.pipeline(place.code(), &.{ lab.big.handle, lab.out.handle }, @sizeOf(MlpPush), .{ .subgroup_size = 64, .spec = &.{ w, nps } });
            defer pipe.deinit(ctx);
            // Correctness first: a few inputs, every network's output against the CPU.
            var check: MlpPush = .{ .iters = 3, .layers = mlp_layers };
            _ = try lab.once(&pipe, &check, mlp_groups);
            try checkMlp(gpa, lab, weights, w, nets, check.iters);

            var push: MlpPush = .{ .iters = 1, .layers = mlp_layers };
            const ns = try lab.measure(&pipe, &push, mlp_groups);
            const flops = @as(f64, @floatFromInt(nets)) * @as(f64, @floatFromInt(push.iters)) * mlp_layers * 2 * @as(f64, @floatFromInt(w * w));
            try report.write(out, .{ .device = "gpu", .probe = "mlp", .variant = variant, .x = @floatFromInt(w), .y = @floatFromInt(nps), .value = flops / ns / 1e3, .unit = "TFLOP/s", .ns = ns, .stats = try statsOf(ctx, &pipe) });
        }
    };
}

/// Read back the network outputs and compare a spread of networks with the CPU twin.
fn checkMlp(gpa: std.mem.Allocator, lab: *Lab, weights: []const f32, w: u32, nets: u32, inputs: u32) !void {
    var host = try gpu.Buffer.init(lab.ctx, @as(u64, nets) * 4, .{ .transfer_dst = true }, .{ .host_visible = true, .host_coherent = true }, .{ .host_cached = true });
    defer host.deinit(lab.ctx);
    try lab.cmds.begin(lab.ctx);
    gpu.barrier(lab.ctx, lab.cmds.cmd, .{ .compute_shader = true }, .{ .shader_storage_write = true }, .{ .copy = true }, .{ .transfer_read = true });
    lab.ctx.vkd.cmdCopyBuffer(lab.cmds.cmd, lab.out.handle, host.handle, &.{.{ .src_offset = 0, .dst_offset = 0, .size = @as(u64, nets) * 4 }});
    gpu.barrier(lab.ctx, lab.cmds.cmd, .{ .copy = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
    try lab.cmds.submitAndWait(lab.ctx);
    const got = host.slice(f32);
    const step = @max(1, nets / 61);
    var net: u32 = 0;
    while (net < nets) : (net += step) {
        const want = try cpuNetwork(gpa, weights[@as(usize, net) * w * w ..][0 .. w * w], w, inputs);
        const err = @abs(got[net] - want);
        if (!(err <= 1e-4 * @as(f32, @floatFromInt(inputs)))) {
            std.log.err("mlp W={d}, network {d}: GPU {d}, CPU {d}", .{ w, net, got[net], want });
            return error.MlpMismatch;
        }
    }
}

/// CPU twin of mlp.glsl for one network: sum over inputs of output neuron 0.
fn cpuNetwork(gpa: std.mem.Allocator, wt: []const f32, w: u32, inputs: u32) !f32 {
    const x = try gpa.alloc(f32, w);
    defer gpa.free(x);
    const y = try gpa.alloc(f32, w);
    defer gpa.free(y);
    var total: f32 = 0;
    for (0..inputs) |n| {
        for (x, 0..) |*v, k| v.* = inputValue(@intCast(n), @intCast(k));
        for (0..mlp_layers) |_| {
            for (y, 0..) |*v, row| {
                var acc: f32 = 0;
                for (0..w) |k| acc = @mulAdd(f32, wt[row * w + k], x[k], acc);
                v.* = std.math.tanh(acc);
            }
            @memcpy(x, y);
        }
        total += x[0];
    }
    return total;
}

pub fn inputValue(n: u32, k: u32) f32 {
    const h = (n *% 2654435761) +% (k *% 40503);
    return @as(f32, @floatFromInt(h >> 24)) * (1.0 / 256.0) - 0.5;
}

// ---------------------------------------------------------------- main

/// The network shader with weights in registers, for the crossover tasks.
pub fn mlpRegCode() []align(4) const u8 {
    return &spv.mlp_reg;
}

/// The fp32 ALU probe shader, for other parts that need a trivial dispatch.
pub fn fma32Code() []align(4) const u8 {
    return &spv.fma32;
}

pub const all = [_][]const u8{ "alu", "lanes", "occupancy", "lds", "stream", "chase", "coop", "mlp" };

pub fn run(gpa: std.mem.Allocator, lab: *Lab, out: *std.Io.Writer, name: []const u8) !bool {
    if (std.mem.eql(u8, name, "alu")) try probeAlu(lab, out) //
    else if (std.mem.eql(u8, name, "lanes")) try probeLanes(lab, out) //
    else if (std.mem.eql(u8, name, "occupancy")) try probeOccupancy(lab, out) //
    else if (std.mem.eql(u8, name, "lds")) try probeLds(lab, out) //
    else if (std.mem.eql(u8, name, "stream")) try probeStream(lab, out) //
    else if (std.mem.eql(u8, name, "chase")) try probeChase(gpa, lab, out) //
    else if (std.mem.eql(u8, name, "coop")) try probeCoop(gpa, lab, out) //
    else if (std.mem.eql(u8, name, "mlp")) try probeMlp(gpa, lab, out) //
    else return false;
    return true;
}

/// Every compile statistic the driver offers for one probe shader, by name and description.
pub fn listStats(lab: *Lab) !void {
    var pipe = try lab.pipeline(&spv.fma32, &.{lab.out.handle}, @sizeOf(FmaPush), .{ .subgroup_size = 64, .spec = &.{ 8, 1 } });
    defer pipe.deinit(lab.ctx);
    var stats: [64]vk.PipelineExecutableStatisticKHR = undefined;
    for (try pipe.statistics(lab.ctx, &stats)) |st| std.debug.print("{s}: {s}\n", .{ std.mem.sliceTo(&st.name, 0), std.mem.sliceTo(&st.description, 0) });
}

test "gpu every probe runs and the networks match the CPU (small sizes)" {
    report.progress = false;
    var feats: Features = .{};
    var ctx = try gpu.Context.init(std.testing.allocator, feats.options(true));
    defer ctx.deinit();
    var lab = try Lab.init(&ctx, true);
    defer lab.deinit();
    var sink: std.Io.Writer.Discarding = .init(&.{});
    for (all) |name| try std.testing.expect(try run(std.testing.allocator, &lab, &sink.writer, name));
    try std.testing.expectEqual(@as(u32, 0), gpu.validation_messages.load(.monotonic));
}

test "gpu stream reads the blocks in strict sweep order" {
    report.progress = false;
    var feats: Features = .{};
    var ctx = try gpu.Context.init(std.testing.allocator, feats.options(true));
    defer ctx.deinit();
    var lab = try Lab.init(&ctx, true);
    defer lab.deinit();
    // Not a power of two, and not a multiple of the grid, so the wrap-around is exercised.
    const nblocks = 48;
    const groups = 20;
    const iters = 7;
    const usage: vk.BufferUsageFlags = .{ .storage_buffer = true };
    const host: vk.MemoryPropertyFlags = .{ .host_visible = true, .host_coherent = true };
    var in = try gpu.Buffer.init(&ctx, nblocks * 4096, usage, host, .{});
    defer in.deinit(&ctx);
    var res = try gpu.Buffer.init(&ctx, groups * 256 * 4, usage, host, .{});
    defer res.deinit(&ctx);
    // vec4 k holds (k, 0, 0, 0): each lane's sum names the blocks it read.
    const v = in.slice(f32);
    @memset(v, 0);
    for (0..nblocks * 256) |k| v[4 * k] = @floatFromInt(k);
    var pipe = try lab.pipeline(&spv.stream, &.{ in.handle, res.handle }, @sizeOf(StreamPush), .{ .subgroup_size = 64 });
    defer pipe.deinit(lab.ctx);
    const push: StreamPush = .{ .nblocks = nblocks, .iters = iters };
    _ = try lab.once(&pipe, &push, groups);
    const got = res.slice(f32);
    for (0..groups) |g| {
        for (0..256) |l| {
            var b: u64 = g % nblocks;
            var want: u64 = 0;
            for (0..iters) |_| {
                want += b * 256 + l;
                b = (b + groups) % nblocks;
            }
            try std.testing.expectEqual(@as(f32, @floatFromInt(want)), got[g * 256 + l]);
        }
    }
    try std.testing.expectEqual(@as(u32, 0), gpu.validation_messages.load(.monotonic));
}

test "gpu ring visits every line once" {
    var ring: [32 * 50]u32 = undefined;
    var rng = std.Random.DefaultPrng.init(1);
    buildRing(&ring, rng.random());
    var seen: [50]bool = @splat(false);
    var p: u32 = 0;
    for (0..50) |_| {
        try std.testing.expect(!seen[p / 32]);
        seen[p / 32] = true;
        p = ring[p];
    }
    try std.testing.expectEqual(@as(u32, 0), p);
}
