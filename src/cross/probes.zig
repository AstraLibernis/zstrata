// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Crossovers: the same task, at growing sizes, on one CPU core, on all CPU threads, and on
//! the GPU, to see where each kind of work belongs.
//!
//!   map         y = a*x + b over n floats                       simple, memory-bound
//!   nets        N networks (64 wide, 4 layers, 16 inputs each)  uniform, compute-bound
//!   trees       128 decision trees (depth 8) over R rows         branchy, scattered reads
//!   chains      M independent 1,000-step chains                 sequential per item, many items
//!   chain-long  one chain of S steps                             purely sequential
//!
//! GPU times come three ways: `gpu-kernel` (GPU timestamps around the dispatch: the GPU's own
//! work), `gpu-resident` (submit, run, wait: inputs already on the GPU, result left there),
//! `gpu-with-transfer` (the CPU writes the inputs into VRAM, the GPU runs, the result is
//! copied back and read: the whole trip a program pays). Every size checks the GPU's result
//! against the CPU's (equal bits where the arithmetic allows it).
//!
//! Each value is µs per run: the median of 5 batches, each batch repeated until it lasts
//! ~10 ms.

const std = @import("std");
const gpu = @import("../gpu.zig");
const vk = gpu.vk;
const report = @import("../report.zig");
const clock = @import("../cpu/probes.zig");
const cpu_host = @import("../cpu/host.zig");
const gpu_probes = @import("../gpu/probes.zig");
const Lab = gpu_probes.Lab;
const Pool = @import("../cpu/pool.zig").Pool;
const mem = @import("../lib/mem.zig");

const spv = struct {
    const saxpy align(@alignOf(u32)) = @embedFile("saxpy.spv").*;
    const trees align(@alignOf(u32)) = @embedFile("trees.spv").*;
    const chains align(@alignOf(u32)) = @embedFile("chains.spv").*;
};

const batch_ns = 10e6;
const reps = 5;

/// µs per call of `r.run()`: median over `reps` batches, each repeated to ~batch_ns.
fn timeIt(r: anytype, quick: bool) !f64 {
    if (quick) {
        const t0 = clock.nowNs();
        try r.run();
        return @as(f64, @floatFromInt(clock.nowNs() - t0)) / 1e3;
    }
    var k: u64 = 1;
    while (true) {
        const t0 = clock.nowNs();
        for (0..k) |_| try r.run();
        const t: f64 = @floatFromInt(clock.nowNs() - t0);
        if (t >= batch_ns or k >= 1 << 20) break;
        k = @max(k * 2, @as(u64, @intFromFloat(@as(f64, @floatFromInt(k)) * batch_ns / @max(t, 1))));
    }
    var xs: [reps]f64 = undefined;
    for (&xs) |*x| {
        const t0 = clock.nowNs();
        for (0..k) |_| try r.run();
        x.* = @as(f64, @floatFromInt(clock.nowNs() - t0)) / @as(f64, @floatFromInt(k));
    }
    return report.median(&xs) / 1e3;
}

/// GPU kernel time alone, µs: median of `reps` dispatches timed by GPU timestamps.
fn kernelTime(lab: *Lab, pipe: *const gpu.ComputePipeline, push: *const anyopaque, groups: u32, quick: bool) !f64 {
    var xs: [reps]f64 = undefined;
    const n: usize = if (quick) 1 else reps;
    for (xs[0..n]) |*x| x.* = try lab.once(pipe, push, groups);
    return report.median(xs[0..n]) / 1e3;
}

const dl_hv: vk.MemoryPropertyFlags = .{ .device_local = true, .host_visible = true, .host_coherent = true };
const usage: vk.BufferUsageFlags = .{ .storage_buffer = true, .transfer_src = true, .transfer_dst = true };

/// Dispatch + optional copy of `out_bytes` back into `host`, then wait. Used by gpu-resident
/// (out_bytes = 0) and gpu-with-transfer.
fn submit(lab: *Lab, pipe: *const gpu.ComputePipeline, push: *const anyopaque, groups: u32, dev_out: vk.Buffer, host: ?*gpu.Buffer, out_bytes: u64) !void {
    const ctx = lab.ctx;
    const cmd = lab.cmds.cmd;
    try lab.cmds.begin(ctx);
    pipe.dispatch(ctx, cmd, push, .{ groups, 1, 1 });
    if (host) |h| {
        gpu.barrier(ctx, cmd, .{ .compute_shader = true }, .{ .shader_storage_write = true }, .{ .copy = true }, .{ .transfer_read = true });
        ctx.vkd.cmdCopyBuffer(cmd, dev_out, h.handle, &.{.{ .src_offset = 0, .dst_offset = 0, .size = out_bytes }});
        gpu.barrier(ctx, cmd, .{ .copy = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
    }
    try lab.cmds.submitAndWait(ctx);
}

fn row(out: *std.Io.Writer, task: []const u8, variant: []const u8, x: u64, us: f64) !void {
    try report.write(out, .{ .device = "cross", .probe = task, .variant = variant, .x = @floatFromInt(x), .value = us, .unit = "us", .ns = us * 1e3 });
}

fn ceilDiv(a: u64, b: u64) u32 {
    return @intCast((a + b - 1) / b);
}

/// Everything the tasks share: the pool, host-visible VRAM for inputs, VRAM for outputs, a
/// cached CPU-side buffer for read-back.
const Rig = struct {
    lab: *Lab,
    pool: *Pool,
    input: gpu.Buffer, // device-local, CPU-writable (resizable BAR)
    output: gpu.Buffer, // device-local
    readback: gpu.Buffer, // CPU memory, cached
    quick: bool,

    const in_bytes = 512 << 20;
    const out_bytes = 256 << 20;

    fn init(gpa: std.mem.Allocator, lab: *Lab, quick: bool) !Rig {
        const ctx = lab.ctx;
        const pool = try Pool.init(gpa, clock.cpuCount());
        errdefer pool.deinit();
        const ib: u64 = if (quick) 16 << 20 else in_bytes;
        const ob: u64 = if (quick) 16 << 20 else out_bytes;
        var input = try gpu.Buffer.init(ctx, ib, usage, dl_hv, .{});
        errdefer input.deinit(ctx);
        var output = try gpu.Buffer.init(ctx, ob, usage, .{ .device_local = true }, .{});
        errdefer output.deinit(ctx);
        const readback = try gpu.Buffer.init(ctx, ob, .{ .transfer_dst = true }, .{ .host_visible = true, .host_coherent = true }, .{ .host_cached = true });
        return .{ .lab = lab, .pool = pool, .input = input, .output = output, .readback = readback, .quick = quick };
    }

    fn deinit(self: *Rig) void {
        const ctx = self.lab.ctx;
        self.readback.deinit(ctx);
        self.output.deinit(ctx);
        self.input.deinit(ctx);
        self.pool.deinit();
        self.* = undefined;
    }

    fn inputSlice(self: *Rig, comptime T: type, n: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(self.input.mapped.?)))[0..n];
    }

    fn readbackSlice(self: *Rig, comptime T: type, n: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(self.readback.mapped.?)))[0..n];
    }
};

// ---------------------------------------------------------------- map

const map_a: f32 = 1.5;
const map_b: f32 = -0.25;

const MapCtx = struct { x: []const f32, y: []f32 };

fn mapRange(ctx: *const MapCtx, begin: usize, end: usize) void {
    const V = @Vector(16, f32);
    const a: V = @splat(map_a);
    const b: V = @splat(map_b);
    var i = begin;
    while (i + 16 <= end) : (i += 16) ctx.y[i..][0..16].* = cpu_host.mulAdd(V, a, ctx.x[i..][0..16].*, b);
    while (i < end) : (i += 1) ctx.y[i] = cpu_host.mulAdd(f32, map_a, ctx.x[i], map_b);
}

fn mapTask(c: *anyopaque, _: usize, begin: usize, end: usize) void {
    mapRange(@ptrCast(@alignCast(c)), begin, end);
}

fn probeMap(gpa: std.mem.Allocator, rig: *Rig, out: *std.Io.Writer) !void {
    const lab = rig.lab;
    const ns: []const u64 = if (rig.quick) &.{ 1 << 10, 64 << 10 } else &.{ 1 << 10, 4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20 };
    const max = ns[ns.len - 1];
    // Page-aligned, 2 MiB pages on Linux (lib/mem.zig): the CPU side reads and writes these 64
    // bytes at a time.
    const x = try mem.allocBig(f32, max, .huge);
    defer mem.freeBig(f32, x);
    const y = try mem.allocBig(f32, max, .huge);
    defer mem.freeBig(f32, y);
    for (x, 0..) |*v, i| v.* = @floatFromInt(i % 1000);
    var pipe = try lab.pipeline(&spv.saxpy, &.{ rig.input.handle, rig.output.handle }, 12, .{ .subgroup_size = 64 });
    defer pipe.deinit(lab.ctx);
    const Push = extern struct { n: u32, a: f32, b: f32 };

    for (ns) |n| {
        const ctx: MapCtx = .{ .x = x[0..n], .y = y[0..n] };
        const push: Push = .{ .n = @intCast(n), .a = map_a, .b = map_b };
        const groups = ceilDiv(n, 256);
        const Cpu1 = struct {
            c: *const MapCtx,
            fn run(s: @This()) !void {
                mapRange(s.c, 0, s.c.x.len);
            }
        };
        const CpuAll = struct {
            c: *const MapCtx,
            p: *Pool,
            fn run(s: @This()) !void {
                s.p.parallelFor(s.c.x.len, @ptrCast(@constCast(s.c)), mapTask, 16 << 10);
            }
        };
        const Resident = struct {
            r: *Rig,
            pp: *const gpu.ComputePipeline,
            push: *const Push,
            g: u32,
            fn run(s: @This()) !void {
                try submit(s.r.lab, s.pp, s.push, s.g, s.r.output.handle, null, 0);
            }
        };
        const Transfer = struct {
            r: *Rig,
            pp: *const gpu.ComputePipeline,
            push: *const Push,
            g: u32,
            x: []const f32,
            y: []f32,
            fn run(s: @This()) !void {
                @memcpy(s.r.inputSlice(f32, s.x.len), s.x);
                try submit(s.r.lab, s.pp, s.push, s.g, s.r.output.handle, &s.r.readback, s.x.len * 4);
                @memcpy(s.y, s.r.readbackSlice(f32, s.y.len));
            }
        };
        try row(out, "map", "cpu-1", n, try timeIt(Cpu1{ .c = &ctx }, rig.quick));
        try row(out, "map", "cpu-all", n, try timeIt(CpuAll{ .c = &ctx, .p = rig.pool }, rig.quick));
        const want = try gpa.dupe(f32, y[0..n]);
        defer gpa.free(want);
        const tr: Transfer = .{ .r = rig, .pp = &pipe, .push = &push, .g = groups, .x = x[0..n], .y = y[0..n] };
        try row(out, "map", "gpu-with-transfer", n, try timeIt(tr, rig.quick));
        if (!std.mem.eql(f32, want, y[0..n])) return error.MapMismatch;
        try row(out, "map", "gpu-resident", n, try timeIt(Resident{ .r = rig, .pp = &pipe, .push = &push, .g = groups }, rig.quick));
        try row(out, "map", "gpu-kernel", n, try kernelTime(lab, &pipe, &push, groups, rig.quick));
    }
}

// ---------------------------------------------------------------- trees

const tree_depth = 8;
const tree_count = 128;
const features = 32;
const inner_nodes = (1 << tree_depth) - 1;

const Forest = struct {
    feat: []u32,
    thr: []f32,
    leaf: []f32,
};

fn treePredict(f: *const Forest, x: []const f32) f32 {
    var sum: f32 = 0;
    for (0..tree_count) |t| {
        var node: u32 = 0;
        for (0..tree_depth) |_| {
            const k = t * inner_nodes + node;
            node = 2 * node + @as(u32, if (x[f.feat[k]] > f.thr[k]) 2 else 1);
        }
        sum += f.leaf[t * (inner_nodes + 1) + (node - inner_nodes)];
    }
    return sum;
}

const TreeCtx = struct { f: *const Forest, x: []const f32, y: []f32 };

fn treeRange(c: *const TreeCtx, begin: usize, end: usize) void {
    for (begin..end) |r| c.y[r] = treePredict(c.f, c.x[r * features ..][0..features]);
}

fn treeTask(c: *anyopaque, _: usize, begin: usize, end: usize) void {
    treeRange(@ptrCast(@alignCast(c)), begin, end);
}

fn probeTrees(gpa: std.mem.Allocator, rig: *Rig, out: *std.Io.Writer) !void {
    const lab = rig.lab;
    const ctx = lab.ctx;
    const rs: []const u64 = if (rig.quick) &.{ 256, 4096 } else &.{ 256, 1 << 10, 4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20, 2 << 20 };
    const max = rs[rs.len - 1];
    var rng = std.Random.DefaultPrng.init(11);
    const rand = rng.random();
    const x = try mem.allocBig(f32, max * features, .huge);
    defer mem.freeBig(f32, x);
    for (x) |*v| v.* = rand.float(f32);
    const y = try mem.allocBig(f32, max, .huge);
    defer mem.freeBig(f32, y);

    // The forest lives on the GPU for the whole probe (a trained model): uploaded once.
    var fbuf = try gpu.Buffer.init(ctx, tree_count * inner_nodes * 4, usage, dl_hv, .{});
    defer fbuf.deinit(ctx);
    var tbuf = try gpu.Buffer.init(ctx, tree_count * inner_nodes * 4, usage, dl_hv, .{});
    defer tbuf.deinit(ctx);
    var lbuf = try gpu.Buffer.init(ctx, tree_count * (inner_nodes + 1) * 4, usage, dl_hv, .{});
    defer lbuf.deinit(ctx);
    const forest: Forest = .{ .feat = fbuf.slice(u32), .thr = tbuf.slice(f32), .leaf = lbuf.slice(f32) };
    // Host-side copies for the CPU (reading the BAR-mapped VRAM from the CPU would be slow).
    const host_feat = try gpa.alloc(u32, forest.feat.len);
    defer gpa.free(host_feat);
    const host_thr = try gpa.alloc(f32, forest.thr.len);
    defer gpa.free(host_thr);
    const host_leaf = try gpa.alloc(f32, forest.leaf.len);
    defer gpa.free(host_leaf);
    for (host_feat) |*v| v.* = rand.uintLessThan(u32, features);
    for (host_thr) |*v| v.* = rand.float(f32);
    for (host_leaf) |*v| v.* = rand.float(f32) - 0.5;
    @memcpy(forest.feat, host_feat);
    @memcpy(forest.thr, host_thr);
    @memcpy(forest.leaf, host_leaf);
    const host_forest: Forest = .{ .feat = host_feat, .thr = host_thr, .leaf = host_leaf };

    var pipe = try lab.pipeline(&spv.trees, &.{ rig.input.handle, fbuf.handle, tbuf.handle, lbuf.handle, rig.output.handle }, 8, .{ .subgroup_size = 64, .spec = &.{ tree_depth, features } });
    defer pipe.deinit(ctx);
    const Push = extern struct { rows: u32, trees: u32 };

    for (rs) |r| {
        const tc: TreeCtx = .{ .f = &host_forest, .x = x[0 .. r * features], .y = y[0..r] };
        const push: Push = .{ .rows = @intCast(r), .trees = tree_count };
        const groups = ceilDiv(r, 64);
        const Cpu1 = struct {
            c: *const TreeCtx,
            fn run(s: @This()) !void {
                treeRange(s.c, 0, s.c.y.len);
            }
        };
        const CpuAll = struct {
            c: *const TreeCtx,
            p: *Pool,
            fn run(s: @This()) !void {
                s.p.parallelFor(s.c.y.len, @ptrCast(@constCast(s.c)), treeTask, 64);
            }
        };
        const Resident = struct {
            rg: *Rig,
            pp: *const gpu.ComputePipeline,
            push: *const Push,
            g: u32,
            fn run(s: @This()) !void {
                try submit(s.rg.lab, s.pp, s.push, s.g, s.rg.output.handle, null, 0);
            }
        };
        const Transfer = struct {
            rg: *Rig,
            pp: *const gpu.ComputePipeline,
            push: *const Push,
            g: u32,
            x: []const f32,
            y: []f32,
            fn run(s: @This()) !void {
                @memcpy(s.rg.inputSlice(f32, s.x.len), s.x);
                try submit(s.rg.lab, s.pp, s.push, s.g, s.rg.output.handle, &s.rg.readback, s.y.len * 4);
                @memcpy(s.y, s.rg.readbackSlice(f32, s.y.len));
            }
        };
        try row(out, "trees", "cpu-1", r, try timeIt(Cpu1{ .c = &tc }, rig.quick));
        try row(out, "trees", "cpu-all", r, try timeIt(CpuAll{ .c = &tc, .p = rig.pool }, rig.quick));
        const want = try gpa.dupe(f32, y[0..r]);
        defer gpa.free(want);
        try row(out, "trees", "gpu-with-transfer", r, try timeIt(Transfer{ .rg = rig, .pp = &pipe, .push = &push, .g = groups, .x = x[0 .. r * features], .y = y[0..r] }, rig.quick));
        if (!std.mem.eql(f32, want, y[0..r])) return error.TreesMismatch;
        try row(out, "trees", "gpu-resident", r, try timeIt(Resident{ .rg = rig, .pp = &pipe, .push = &push, .g = groups }, rig.quick));
        try row(out, "trees", "gpu-kernel", r, try kernelTime(lab, &pipe, &push, groups, rig.quick));
    }
}

// ---------------------------------------------------------------- chains

const chain_steps = 1000;

/// The logistic map from chains.comp, 16 chains at a time (plain multiply and subtract, no
/// fused multiply-add, so the bits match the GPU's `precise` arithmetic).
fn chainRange(y: []f32, begin: usize, end: usize, steps: u32) void {
    const V = @Vector(16, f32);
    var c = begin;
    while (c < end) : (c += 16) {
        var x: V = undefined;
        inline for (0..16) |l| x[l] = chainStart(c + l);
        for (0..steps) |_| {
            const t = @as(V, @splat(3.9)) * x;
            x = t * (@as(V, @splat(1.0)) - x);
        }
        inline for (0..16) |l| {
            if (c + l < end) y[c + l] = x[l];
        }
    }
}

fn chainStart(c: usize) f32 {
    return 0.1 + 0.8 * @as(f32, @floatFromInt(c % 1024)) / 1024.0;
}

const ChainCtx = struct { y: []f32, steps: u32 };

fn chainTask(c: *anyopaque, _: usize, begin: usize, end: usize) void {
    const ctx: *const ChainCtx = @ptrCast(@alignCast(c));
    chainRange(ctx.y, begin, end, ctx.steps);
}

fn probeChains(gpa: std.mem.Allocator, rig: *Rig, out: *std.Io.Writer, long: bool) !void {
    const lab = rig.lab;
    const task = if (long) "chain-long" else "chains";
    const xs: []const u64 = if (rig.quick)
        &.{ 1, 64 }
    else if (long)
        &.{ 1_000, 10_000, 100_000, 1_000_000, 10_000_000 }
    else
        &.{ 1, 16, 256, 4 << 10, 64 << 10, 1 << 20 };
    const max_chains: usize = if (long) 16 else xs[xs.len - 1] + 16;
    const y = try gpa.alloc(f32, max_chains);
    defer gpa.free(y);
    var pipe = try lab.pipeline(&spv.chains, &.{rig.output.handle}, 8, .{ .subgroup_size = 64 });
    defer pipe.deinit(lab.ctx);
    const Push = extern struct { chains: u32, steps: u32 };

    for (xs) |v| {
        const m: usize = if (long) 1 else v;
        const steps: u32 = if (long) @intCast(v) else chain_steps;
        const cc: ChainCtx = .{ .y = y[0..m], .steps = steps };
        const push: Push = .{ .chains = @intCast(m), .steps = steps };
        const groups = ceilDiv(m, 64);
        const Cpu1 = struct {
            c: *const ChainCtx,
            fn run(s: @This()) !void {
                chainRange(s.c.y, 0, s.c.y.len, s.c.steps);
            }
        };
        const CpuAll = struct {
            c: *const ChainCtx,
            p: *Pool,
            fn run(s: @This()) !void {
                s.p.parallelFor(s.c.y.len, @ptrCast(@constCast(s.c)), chainTask, 16);
            }
        };
        const Resident = struct {
            rg: *Rig,
            pp: *const gpu.ComputePipeline,
            push: *const Push,
            g: u32,
            fn run(s: @This()) !void {
                try submit(s.rg.lab, s.pp, s.push, s.g, s.rg.output.handle, null, 0);
            }
        };
        const Transfer = struct {
            rg: *Rig,
            pp: *const gpu.ComputePipeline,
            push: *const Push,
            g: u32,
            y: []f32,
            fn run(s: @This()) !void {
                try submit(s.rg.lab, s.pp, s.push, s.g, s.rg.output.handle, &s.rg.readback, s.y.len * 4);
                @memcpy(s.y, s.rg.readbackSlice(f32, s.y.len));
            }
        };
        try row(out, task, "cpu-1", v, try timeIt(Cpu1{ .c = &cc }, rig.quick));
        if (!long) try row(out, task, "cpu-all", v, try timeIt(CpuAll{ .c = &cc, .p = rig.pool }, rig.quick));
        const want = try gpa.dupe(f32, y[0..m]);
        defer gpa.free(want);
        try row(out, task, "gpu-with-transfer", v, try timeIt(Transfer{ .rg = rig, .pp = &pipe, .push = &push, .g = groups, .y = y[0..m] }, rig.quick));
        if (!std.mem.eql(f32, want, y[0..m])) return error.ChainsMismatch;
        try row(out, task, "gpu-resident", v, try timeIt(Resident{ .rg = rig, .pp = &pipe, .push = &push, .g = groups }, rig.quick));
        try row(out, task, "gpu-kernel", v, try kernelTime(lab, &pipe, &push, groups, rig.quick));
    }
}

// ---------------------------------------------------------------- nets

const net_w = 64;
const net_layers = 4;
const net_inputs = 16;

/// CPU twin of mlp.glsl at W = 64, one network: sum over inputs of output neuron 0. Rows are
/// dot products of 4 x 16-wide vectors.
fn netForward(w: []const f32) f32 {
    const V = @Vector(16, f32);
    var x: [net_w]f32 align(64) = undefined;
    var y: [net_w]f32 align(64) = undefined;
    var total: f32 = 0;
    for (0..net_inputs) |n| {
        for (&x, 0..) |*v, k| v.* = gpu_probes.inputValue(@intCast(n), @intCast(k));
        for (0..net_layers) |_| {
            for (0..net_w) |r| {
                var acc: V = @splat(0);
                inline for (0..net_w / 16) |q| acc = cpu_host.mulAdd(V, w[r * net_w + q * 16 ..][0..16].*, x[q * 16 ..][0..16].*, acc);
                y[r] = std.math.tanh(@reduce(.Add, acc));
            }
            x = y;
        }
        total += x[0];
    }
    return total;
}

const NetCtx = struct { w: []const f32, y: []f32 };

fn netRange(c: *const NetCtx, begin: usize, end: usize) void {
    for (begin..end) |i| c.y[i] = netForward(c.w[i * net_w * net_w ..][0 .. net_w * net_w]);
}

fn netTask(c: *anyopaque, _: usize, begin: usize, end: usize) void {
    netRange(@ptrCast(@alignCast(c)), begin, end);
}

fn probeNets(gpa: std.mem.Allocator, rig: *Rig, out: *std.Io.Writer) !void {
    const lab = rig.lab;
    const ns: []const u64 = if (rig.quick) &.{ 1, 16 } else &.{ 1, 4, 16, 64, 256, 1024, 4096 };
    const max = ns[ns.len - 1];
    var rng = std.Random.DefaultPrng.init(5);
    const w = try gpa.alloc(f32, max * net_w * net_w);
    defer gpa.free(w);
    for (w) |*v| v.* = (rng.random().float(f32) * 2 - 1) / 16.0;
    const y = try gpa.alloc(f32, max);
    defer gpa.free(y);
    var pipe = try lab.pipeline(gpu_probes.mlpRegCode(), &.{ rig.input.handle, rig.output.handle }, 8, .{ .subgroup_size = 64, .spec = &.{ net_w, 1 } });
    defer pipe.deinit(lab.ctx);
    const Push = extern struct { iters: u32, layers: u32 };

    for (ns) |n| {
        const nc: NetCtx = .{ .w = w[0 .. n * net_w * net_w], .y = y[0..n] };
        const push: Push = .{ .iters = net_inputs, .layers = net_layers };
        const groups: u32 = @intCast(n);
        const Cpu1 = struct {
            c: *const NetCtx,
            fn run(s: @This()) !void {
                netRange(s.c, 0, s.c.y.len);
            }
        };
        const CpuAll = struct {
            c: *const NetCtx,
            p: *Pool,
            fn run(s: @This()) !void {
                s.p.parallelFor(s.c.y.len, @ptrCast(@constCast(s.c)), netTask, 1);
            }
        };
        const Resident = struct {
            rg: *Rig,
            pp: *const gpu.ComputePipeline,
            push: *const Push,
            g: u32,
            fn run(s: @This()) !void {
                try submit(s.rg.lab, s.pp, s.push, s.g, s.rg.output.handle, null, 0);
            }
        };
        const Transfer = struct {
            rg: *Rig,
            pp: *const gpu.ComputePipeline,
            push: *const Push,
            g: u32,
            w: []const f32,
            y: []f32,
            fn run(s: @This()) !void {
                @memcpy(s.rg.inputSlice(f32, s.w.len), s.w);
                try submit(s.rg.lab, s.pp, s.push, s.g, s.rg.output.handle, &s.rg.readback, s.y.len * 4);
                @memcpy(s.y, s.rg.readbackSlice(f32, s.y.len));
            }
        };
        try row(out, "nets", "cpu-1", n, try timeIt(Cpu1{ .c = &nc }, rig.quick));
        try row(out, "nets", "cpu-all", n, try timeIt(CpuAll{ .c = &nc, .p = rig.pool }, rig.quick));
        const want = try gpa.dupe(f32, y[0..n]);
        defer gpa.free(want);
        try row(out, "nets", "gpu-with-transfer", n, try timeIt(Transfer{ .rg = rig, .pp = &pipe, .push = &push, .g = groups, .w = nc.w, .y = y[0..n] }, rig.quick));
        // Different summation order and tanh on each side: compare within a tolerance.
        for (want, y[0..n]) |a, b| if (!(@abs(a - b) <= 1e-4 * net_inputs)) return error.NetsMismatch;
        try row(out, "nets", "gpu-resident", n, try timeIt(Resident{ .rg = rig, .pp = &pipe, .push = &push, .g = groups }, rig.quick));
        try row(out, "nets", "gpu-kernel", n, try kernelTime(lab, &pipe, &push, groups, rig.quick));
    }
}

pub const all = [_][]const u8{ "map", "nets", "trees", "chains", "chain-long" };

pub fn run(gpa: std.mem.Allocator, lab: *Lab, out: *std.Io.Writer, name: []const u8, quick: bool) !bool {
    var known = false;
    for (all) |a| known = known or std.mem.eql(u8, a, name);
    if (!known) return false;
    var rig = try Rig.init(gpa, lab, quick);
    defer rig.deinit();
    if (std.mem.eql(u8, name, "map")) try probeMap(gpa, &rig, out) //
    else if (std.mem.eql(u8, name, "nets")) try probeNets(gpa, &rig, out) //
    else if (std.mem.eql(u8, name, "trees")) try probeTrees(gpa, &rig, out) //
    else if (std.mem.eql(u8, name, "chains")) try probeChains(gpa, &rig, out, false) //
    else try probeChains(gpa, &rig, out, true);
    return true;
}

test "cpu twins of the crossover tasks agree with plain loops" {
    var y: [40]f32 = undefined;
    chainRange(&y, 0, 37, 5);
    for (0..37) |c| {
        var x = chainStart(c);
        for (0..5) |_| {
            const t = 3.9 * x;
            x = t * (1.0 - x);
        }
        try std.testing.expectEqual(x, y[c]);
    }
}

test "cross every task runs and the GPU matches the CPU (small sizes)" {
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
