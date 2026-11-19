// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! The CPU side of the map, the same questions as the GPU probes:
//!
//!   clock   the core clock, from a chain of dependent one-cycle adds: one core busy, all busy
//!   alu    multiply-adds by vector width (scalar, SSE, AVX2, AVX-512) and independent chains
//!           per thread; then by thread count (one per core, then the SMT partners)
//!   chase   load latency by working-set size (L1 -> L2 -> L3 -> RAM)
//!   stream  read bandwidth by working-set size, one thread and all threads
//!   branch  cost per element of a predictable branch, a random one, and no branch
//!   sync    cost of handing work to threads: spawn + join, and one pool round trip
//!   alloc   writing to freshly allocated memory vs memory already touched
//!
//! Every row is the median of 5 runs of ~25 ms, after calibration. Threads are pinned: thread
//! i runs on CPU order[i], which visits each core once before any SMT partner.

const std = @import("std");
const host = @import("host.zig");
const report = @import("../report.zig");
const Pool = @import("pool.zig").Pool;

test {
    _ = @import("pool.zig");
}

const target_ns = 25e6;
const reps = 5;
/// CPU numbers in the order threads use them: one per core, then the second thread of each
/// core. Filled from the OS at start (`initTopology`); this default matches this machine on Linux.
var order: [64]u32 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 } ++ @as([48]u32, @splat(0));
var cpus: u32 = 16;
/// How many entries at the front of `order` are one CPU per core; 0 until the topology is read.
var core_count: u32 = 0;

pub const nowNs = host.nowNs;
const pin = host.pin;

/// Read the core topology from the OS so `order` lists one CPU per core first.
pub fn initTopology(io: std.Io) void {
    var firsts: [64]u32 = undefined;
    var seconds: [64]u32 = undefined;
    var nf: u32 = 0;
    var ns: u32 = 0;
    if (!host.topology(io, &firsts, &nf, &seconds, &ns)) return;
    @memcpy(order[0..nf], firsts[0..nf]);
    @memcpy(order[nf..][0..ns], seconds[0..ns]);
    cpus = nf + ns;
    core_count = nf;
}

pub fn cpuCount() u32 {
    return cpus;
}

/// Wait step for the start barrier: spin briefly (a pinned worker reacts to the start signal at
/// once), then give the CPU away. Pure spinning livelocks under ThreadSanitizer, where every
/// atomic load takes a lock: 16 spinning workers starve the thread that must set the flag.
fn backoff(spins: *u32) void {
    spins.* +%= 1;
    if (spins.* < spin_budget) std.atomic.spinLoopHint() else std.Thread.yield() catch {}; // zsnag:ok — a failed yield just means spin again
}

const spin_budget = 4096;

/// Run `work(ctx, thread, n)` on `threads` pinned threads that start together; returns the
/// wall time from the start signal to the last thread finishing.
fn team(threads: u32, comptime Ctx: type, ctx: *const Ctx, comptime work: fn (*const Ctx, u32, u64) void, n: u64) !f64 {
    const Shared = struct {
        ready: std.atomic.Value(u32) = .init(0),
        go: std.atomic.Value(bool) = .init(false),
        ends: [64]u64 = undefined,
    };
    var sh: Shared = .{};
    const Worker = struct {
        fn run(s: *Shared, c: *const Ctx, t: u32, iters: u64) void {
            pin(order[t]);
            _ = s.ready.fetchAdd(1, .acq_rel);
            var spins: u32 = 0;
            while (!s.go.load(.acquire)) backoff(&spins);
            work(c, t, iters);
            s.ends[t] = nowNs();
        }
    };
    var handles: [64]std.Thread = undefined;
    var spawned: u32 = 0;
    defer for (handles[0..spawned]) |h| h.join();
    while (spawned < threads) : (spawned += 1) {
        handles[spawned] = try std.Thread.spawn(.{}, Worker.run, .{ &sh, ctx, spawned, n });
    }
    var spins: u32 = 0;
    while (sh.ready.load(.acquire) < threads) backoff(&spins);
    const t0 = nowNs();
    sh.go.store(true, .release);
    for (handles[0..spawned]) |h| h.join();
    spawned = 0;
    var last: u64 = t0;
    for (sh.ends[0..threads]) |e| last = @max(last, e);
    return @floatFromInt(last - t0);
}

/// Grow n until one run takes ~target_ns, then the median of `reps` runs. `n` keeps the final
/// value, for the work count.
fn measure(threads: u32, comptime Ctx: type, ctx: *const Ctx, comptime work: fn (*const Ctx, u32, u64) void, n: *u64, quick: bool) !f64 {
    if (quick) return team(threads, Ctx, ctx, work, n.*);
    var t = try team(threads, Ctx, ctx, work, n.*);
    while (t < target_ns / 8) {
        n.* *= if (t < target_ns / 64) 8 else 2;
        t = try team(threads, Ctx, ctx, work, n.*);
    }
    n.* = @intFromFloat(@max(1, @as(f64, @floatFromInt(n.*)) * target_ns / t));
    var xs: [reps]f64 = undefined;
    for (&xs) |*x| x.* = try team(threads, Ctx, ctx, work, n.*);
    return report.median(&xs);
}

// ---------------------------------------------------------------- alu

/// K independent multiply-add chains of V-wide f32 vectors, n rounds. Checked in the ReleaseFast
/// disassembly: one vfmadd213ss/ps per chain on xmm/ymm/zmm by V, scalar chains not merged into
/// vectors, and no stack spills up to 28 chains at V = 16 (AVX-512 has 32 vector registers).
fn fmaChains(comptime V: usize, comptime K: usize, n: u64, seed: f32) f32 {
    const Vec = @Vector(V, f32);
    var acc: [K]Vec = undefined;
    inline for (0..K) |i| acc[i] = @splat(seed + @as(f32, @floatFromInt(i)) * 1e-3);
    const a: Vec = @splat(0.999);
    const b: Vec = @splat(0.001);
    var it: u64 = 0;
    while (it < n) : (it += 1) {
        inline for (0..K) |i| acc[i] = @mulAdd(Vec, acc[i], a, b);
    }
    var s: Vec = @splat(0);
    inline for (0..K) |i| s += acc[i];
    return @reduce(.Add, s);
}

const AluCtx = struct { f: *const fn (u64, f32) f32 };

fn aluWork(ctx: *const AluCtx, t: u32, n: u64) void {
    std.mem.doNotOptimizeAway(ctx.f(n, @floatFromInt(t)));
}

fn aluFn(comptime V: usize, comptime K: usize) *const fn (u64, f32) f32 {
    return struct {
        fn f(n: u64, seed: f32) f32 {
            return fmaChains(V, K, n, seed);
        }
    }.f;
}

// ---------------------------------------------------------------- clock

/// Adds per round of `addChain`.
const clock_adds = 32;

/// `s` written `n` times (assembly text, built at compile time).
fn repeated(comptime s: []const u8, comptime n: usize) []const u8 {
    comptime {
        var r: []const u8 = "";
        for (0..n) |_| r = r ++ s;
        return r;
    }
}

/// `n` rounds of `clock_adds` dependent register adds, the add chain the clock tracker times.
/// A dependent integer add takes one cycle on x86_64 and aarch64 cores, so the chain runs at
/// one add per cycle: adds / ns = GHz. The loop counter runs alongside, off the chain. Returns
/// the sum (n * clock_adds), which the test checks. Null on other architectures.
fn addChain(n: u64) ?u64 {
    if (n == 0) return 0;
    var x: u64 = 0;
    var left = n;
    switch (@import("builtin").cpu.arch) {
        .x86_64 => asm volatile ("1:\n" ++ repeated("addq $1, %[x]\n", clock_adds) ++ "decq %[left]\njnz 1b"
            : [x] "+r" (x),
              [left] "+r" (left),
            :
            : .{ .cc = true }),
        .aarch64 => asm volatile ("1:\n" ++ repeated("add %[x], %[x], #1\n", clock_adds) ++ "subs %[left], %[left], #1\nb.ne 1b"
            : [x] "+r" (x),
              [left] "+r" (left),
            :
            : .{ .nzcv = true }),
        else => return null,
    }
    return x;
}

const ClockCtx = struct {};

fn clockWork(_: *const ClockCtx, _: u32, n: u64) void {
    std.mem.doNotOptimizeAway(addChain(n));
}

/// The clock of each running core in GHz, with `threads` cores busy (one thread per core).
/// Null where `addChain` has no assembly.
fn clockGhz(threads: u32, quick: bool) !?f64 {
    if (addChain(1) == null) return null;
    const ctx: ClockCtx = .{};
    var n: u64 = 1024;
    const ns = try measure(threads, ClockCtx, &ctx, clockWork, &n, quick);
    return @as(f64, @floatFromInt(n * clock_adds)) / ns;
}

/// The clock with one core busy, then with every core busy (one thread each): what the boost
/// gives a single thread and what it gives all-core work.
pub fn probeClock(out: *std.Io.Writer, quick: bool) !void {
    const counts = [_]u32{ 1, @max(1, cores()) };
    for (counts, [_][]const u8{ "1-core", "all-cores" }) |t, name| {
        const ghz = try clockGhz(t, quick) orelse return;
        try report.write(out, .{ .device = "cpu", .probe = "clock", .variant = name, .x = @floatFromInt(t), .value = ghz, .unit = "GHz", .ns = 0 });
    }
}

const widths = [_]usize{ 1, 4, 8, 16 };
const chains = [_]usize{ 1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64 };

fn widthName(v: usize) []const u8 {
    return switch (v) {
        1 => "scalar",
        4 => "sse-4",
        8 => "avx2-8",
        16 => "avx512-16",
        else => unreachable,
    };
}

pub fn probeAlu(out: *std.Io.Writer, quick: bool) !void {
    // Width x chains on one thread.
    inline for (widths) |v| inline for (chains) |k| {
        if (!quick or (k == 1 or k == 16)) {
            const ctx: AluCtx = .{ .f = aluFn(v, k) };
            var n: u64 = 1024;
            const ns = try measure(1, AluCtx, &ctx, aluWork, &n, quick);
            const flops = 2.0 * @as(f64, @floatFromInt(v * k)) * @as(f64, @floatFromInt(n));
            try report.write(out, .{ .device = "cpu", .probe = "alu", .variant = widthName(v), .x = @floatFromInt(k), .value = flops / ns / 1e3, .unit = "TFLOP/s", .ns = ns });
        }
    };
    // Threads, at enough chains to fill each core.
    inline for (.{ 8, 16 }) |v| {
        const ctx: AluCtx = .{ .f = aluFn(v, 16) };
        var t: u32 = 1;
        while (t <= cpus) : (t = if (t < 8) t * 2 else t + 4) {
            if (quick and t > 2) break;
            var n: u64 = 1024;
            const ns = try measure(t, AluCtx, &ctx, aluWork, &n, quick);
            const flops = 2.0 * @as(f64, @floatFromInt(v * 16)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(t));
            const name = if (v == 8) "threads-avx2" else "threads-avx512";
            try report.write(out, .{ .device = "cpu", .probe = "alu", .variant = name, .x = @floatFromInt(t), .value = flops / ns / 1e3, .unit = "TFLOP/s", .ns = ns });
        }
    }
}

// ---------------------------------------------------------------- chase

const line_u32 = 16; // u32s per 64-byte cache line

/// One random cycle through every 64-byte line of `ring` (Sattolo's shuffle): ring[p] holds the
/// index of the next line's first u32.
pub fn buildRing(gpa: std.mem.Allocator, ring: []u32, rand: std.Random, comptime line: usize) !void {
    const n = ring.len / line;
    const ord = try gpa.alloc(u32, n);
    defer gpa.free(ord);
    for (ord, 0..) |*o, i| o.* = @intCast(i);
    var i = n;
    while (i > 1) {
        i -= 1;
        std.mem.swap(u32, &ord[i], &ord[rand.uintLessThan(usize, i)]);
    }
    for (0..n) |k| ring[ord[k] * line] = ord[(k + 1) % n] * @as(u32, line);
}

const ChaseCtx = struct { ring: []const u32, start: *u32 };

fn chaseWork(ctx: *const ChaseCtx, _: u32, n: u64) void {
    var p = ctx.start.*;
    var i: u64 = 0;
    while (i < n) : (i += 1) p = ctx.ring[p];
    // The next run continues where this one stopped, so repeated runs never re-walk the same
    // recent lines (those would come from a cache).
    ctx.start.* = p;
}

fn sizes(buf: []u64, max: u64) []u64 {
    var n: usize = 0;
    var s: u64 = 4 << 10;
    while (s <= max and n + 1 < buf.len) : (s *= 2) {
        buf[n] = s;
        n += 1;
        if (s + s / 2 <= max) {
            buf[n] = s + s / 2;
            n += 1;
        }
    }
    return buf[0..n];
}

pub fn probeChase(gpa: std.mem.Allocator, out: *std.Io.Writer, quick: bool) !void {
    const max: u64 = if (quick) 1 << 20 else 1 << 30;
    var buf: [64]u64 = undefined;
    var rng = std.Random.DefaultPrng.init(7);
    for (sizes(&buf, max)) |s| {
        const ring = try gpa.alloc(u32, s / 4);
        defer gpa.free(ring);
        @memset(ring, 0);
        try buildRing(gpa, ring, rng.random(), line_u32);
        var start: u32 = 0;
        const ctx: ChaseCtx = .{ .ring = ring, .start = &start };
        var n: u64 = 1024;
        const ns = try measure(1, ChaseCtx, &ctx, chaseWork, &n, quick);
        try report.write(out, .{ .device = "cpu", .probe = "chase", .variant = "latency", .x = @floatFromInt(s), .value = ns / @as(f64, @floatFromInt(n)), .unit = "ns", .ns = ns });
    }
}

// ---------------------------------------------------------------- stream

const StreamCtx = struct { data: []const f32, threads: u32 };

/// Each thread sums its own slice of the working set, n passes over it.
fn streamWork(ctx: *const StreamCtx, t: u32, n: u64) void {
    const V = @Vector(16, f32);
    const per = ctx.data.len / ctx.threads / 64 * 64;
    const mine = ctx.data[t * per ..][0..per];
    var a0: V = @splat(0);
    var a1: V = @splat(0);
    var a2: V = @splat(0);
    var a3: V = @splat(0);
    var pass: u64 = 0;
    while (pass < n) : (pass += 1) {
        var i: usize = 0;
        while (i < per) : (i += 64) {
            a0 += mine[i..][0..16].*;
            a1 += mine[i + 16 ..][0..16].*;
            a2 += mine[i + 32 ..][0..16].*;
            a3 += mine[i + 48 ..][0..16].*;
        }
    }
    std.mem.doNotOptimizeAway(@reduce(.Add, a0 + a1 + a2 + a3));
}

pub fn probeStream(gpa: std.mem.Allocator, out: *std.Io.Writer, quick: bool) !void {
    const max: u64 = if (quick) 1 << 20 else 1 << 30;
    const data = try gpa.alloc(f32, max / 4);
    defer gpa.free(data);
    for (data, 0..) |*d, i| d.* = @floatFromInt(i & 1023);
    for ([_]u32{ 1, cpus }) |threads| {
        var s: u64 = 4 << 10;
        while (s <= max) : (s *= 2) {
            if (s / 4 / threads < 64) continue;
            const ctx: StreamCtx = .{ .data = data[0 .. s / 4], .threads = threads };
            var n: u64 = 1;
            const ns = try measure(threads, StreamCtx, &ctx, streamWork, &n, quick);
            const per = s / 4 / threads / 64 * 64;
            const bytes = @as(f64, @floatFromInt(per * 4 * threads)) * @as(f64, @floatFromInt(n));
            try report.write(out, .{ .device = "cpu", .probe = "stream", .variant = if (threads == 1) "read-1-thread" else "read-all-threads", .x = @floatFromInt(s), .y = @floatFromInt(threads), .value = bytes / ns, .unit = "GB/s", .ns = ns });
        }
    }
}

// ---------------------------------------------------------------- branch

const BranchCtx = struct { data: []const u8, mode: enum { branch, branchless } };

fn branchWork(ctx: *const BranchCtx, _: u32, n: u64) void {
    var sum: u32 = 0;
    var pass: u64 = 0;
    while (pass < n) : (pass += 1) {
        switch (ctx.mode) {
            .branch => for (ctx.data) |x| {
                // A different volatile asm marker in each arm forbids LLVM from turning the
                // branch into a select or vectorizing the loop (identical markers get hoisted
                // out of both arms): the CPU must predict a real jump.
                if (x >= 128) {
                    asm volatile ("# taken");
                    sum +%= x;
                } else {
                    asm volatile ("# not taken");
                    sum ^= x;
                }
            },
            .branchless => for (ctx.data) |x| {
                asm volatile ("");
                sum = if (x >= 128) sum +% x else sum ^ x;
            },
        }
    }
    std.mem.doNotOptimizeAway(sum);
}

pub fn probeBranch(gpa: std.mem.Allocator, out: *std.Io.Writer, quick: bool) !void {
    const n_items = 64 << 10; // 64 KiB: stays in L2, so the branch is what is measured
    const random = try gpa.alloc(u8, n_items);
    defer gpa.free(random);
    var rng = std.Random.DefaultPrng.init(3);
    rng.random().bytes(random);
    const sorted = try gpa.dupe(u8, random);
    defer gpa.free(sorted);
    std.mem.sort(u8, sorted, {}, std.sort.asc(u8));
    const Case = struct { name: []const u8, data: []const u8, mode: @FieldType(BranchCtx, "mode") };
    for ([_]Case{
        .{ .name = "predictable", .data = sorted, .mode = .branch },
        .{ .name = "random", .data = random, .mode = .branch },
        .{ .name = "no-branch", .data = random, .mode = .branchless },
    }) |c| {
        const ctx: BranchCtx = .{ .data = c.data, .mode = c.mode };
        var n: u64 = 1;
        const ns = try measure(1, BranchCtx, &ctx, branchWork, &n, quick);
        try report.write(out, .{ .device = "cpu", .probe = "branch", .variant = c.name, .x = 0, .value = ns / (@as(f64, @floatFromInt(n)) * n_items), .unit = "ns/item", .ns = ns });
    }
}

// ---------------------------------------------------------------- sync

pub fn probeSync(gpa: std.mem.Allocator, out: *std.Io.Writer, quick: bool) !void {
    const rounds: u32 = if (quick) 10 else 2000;
    // Spawn + join one thread that does nothing.
    {
        var xs: [reps]f64 = undefined;
        for (&xs) |*x| {
            const t0 = nowNs();
            for (0..rounds / 10) |_| {
                const h = try std.Thread.spawn(.{}, struct {
                    fn f() void {}
                }.f, .{});
                h.join();
            }
            x.* = @as(f64, @floatFromInt(nowNs() - t0)) / @as(f64, @floatFromInt(rounds / 10));
        }
        try report.write(out, .{ .device = "cpu", .probe = "sync", .variant = "spawn-join", .x = 1, .value = report.median(&xs) / 1e3, .unit = "us", .ns = report.median(&xs) });
    }
    // The same with a 1 MiB stack. Zig's default (16 MiB) is committed up front on Windows,
    // which alone costs ~40 us there; Linux maps stacks lazily either way.
    {
        var xs: [reps]f64 = undefined;
        for (&xs) |*x| {
            const t0 = nowNs();
            for (0..rounds / 10) |_| {
                const h = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, struct {
                    fn f() void {}
                }.f, .{});
                h.join();
            }
            x.* = @as(f64, @floatFromInt(nowNs() - t0)) / @as(f64, @floatFromInt(rounds / 10));
        }
        try report.write(out, .{ .device = "cpu", .probe = "sync", .variant = "spawn-join-1mib", .x = 1, .value = report.median(&xs) / 1e3, .unit = "us", .ns = report.median(&xs) });
    }
    // One parallelFor round trip with empty tasks on the pool (pool.zig): back to back (workers still
    // spinning), and after 5 ms idle (workers parked, so they must be woken).
    var t: u32 = 2;
    while (t <= cpus) : (t *= 2) {
        const pool = try Pool.init(gpa, t);
        defer pool.deinit();
        const Empty = struct {
            fn task(_: *anyopaque, _: usize, _: usize, _: usize) void {}
        };
        var dummy: u8 = 0;
        var hot: [reps]f64 = undefined;
        for (&hot) |*x| {
            const t0 = nowNs();
            for (0..rounds) |_| pool.parallelFor(t * 4, &dummy, Empty.task, 1);
            x.* = @as(f64, @floatFromInt(nowNs() - t0)) / @as(f64, @floatFromInt(rounds));
        }
        try report.write(out, .{ .device = "cpu", .probe = "sync", .variant = "pool-hot", .x = @floatFromInt(t), .value = report.median(&hot) / 1e3, .unit = "us", .ns = report.median(&hot) });
        var idle: [reps * 4]f64 = undefined;
        for (&idle) |*x| {
            host.sleepMs(5);
            const t0 = nowNs();
            pool.parallelFor(t * 4, &dummy, Empty.task, 1);
            x.* = @floatFromInt(nowNs() - t0);
        }
        try report.write(out, .{ .device = "cpu", .probe = "sync", .variant = "pool-after-idle", .x = @floatFromInt(t), .value = report.median(&idle) / 1e3, .unit = "us", .ns = report.median(&idle) });
    }
}

// ---------------------------------------------------------------- alloc

pub fn probeAlloc(out: *std.Io.Writer, quick: bool) !void {
    const size: usize = if (quick) 16 << 20 else 1 << 30;
    var first: [reps]f64 = undefined;
    var again: [reps]f64 = undefined;
    for (0..reps) |r| {
        const mem = try std.heap.page_allocator.alloc(u8, size);
        defer std.heap.page_allocator.free(mem);
        var t0 = nowNs();
        @memset(mem, 1);
        first[r] = @floatFromInt(nowNs() - t0);
        t0 = nowNs();
        @memset(mem, 2);
        again[r] = @floatFromInt(nowNs() - t0);
        std.mem.doNotOptimizeAway(mem[size / 2]);
    }
    const f = report.median(&first);
    const a = report.median(&again);
    const sz: f64 = @floatFromInt(size);
    try report.write(out, .{ .device = "cpu", .probe = "alloc", .variant = "first-write", .x = sz, .value = sz / f, .unit = "GB/s", .ns = f });
    try report.write(out, .{ .device = "cpu", .probe = "alloc", .variant = "write-again", .x = sz, .value = sz / a, .unit = "GB/s", .ns = a });
}

// ---------------------------------------------------------------- disturbance check

/// Three quick markers taken at checkpoints during a run (before each part and at the end), so
/// a run disturbed by other work on the machine flags itself instead of being caught by hand:
/// L1 latency (one core's clock and its own cache), RAM latency (the memory path), and
/// AVX-512 FMA on one thread per core (anything else running on any core).
pub const Marks = struct { l1_ns: f64, ram_ns: f64, fma_tflops: f64 };

/// A run is flagged when any marker moves more than this between its best and worst
/// checkpoint. Clean runs move them by at most 6% even across separate runs; a disturbed run
/// moved L1 latency 2.6-3.4x.
pub const disturbed_tolerance = 0.10;

pub const Check = struct {
    l1: []u32,
    ram: []u32,
    l1_at: u32 = 0,
    ram_at: u32 = 0,
    quick: bool,
    marks: std.ArrayList(Marks) = .empty,

    /// RAM ring: 1 GiB, far past the 96 MiB L3 (the 256 MiB chase row is too noisy as a marker,
    /// 12-13% across clean runs). Built once and kept for the whole run.
    pub fn init(gpa: std.mem.Allocator, quick: bool) !Check {
        const ram_bytes: usize = if (quick) 1 << 20 else 1 << 30;
        const l1 = try gpa.alloc(u32, (4 << 10) / 4);
        errdefer gpa.free(l1);
        const ram = try gpa.alloc(u32, ram_bytes / 4);
        errdefer gpa.free(ram);
        var rng = std.Random.DefaultPrng.init(11);
        @memset(l1, 0);
        try buildRing(gpa, l1, rng.random(), line_u32);
        @memset(ram, 0);
        try buildRing(gpa, ram, rng.random(), line_u32);
        return .{ .l1 = l1, .ram = ram, .quick = quick };
    }

    pub fn deinit(self: *Check, gpa: std.mem.Allocator) void {
        gpa.free(self.l1);
        gpa.free(self.ram);
        self.marks.deinit(gpa);
        self.* = undefined;
    }

    /// Take the markers at checkpoint `point` and write them as `cpu,check,<marker>,<point>`.
    pub fn take(self: *Check, gpa: std.mem.Allocator, out: *std.Io.Writer, point: u32) !void {
        const x: f64 = @floatFromInt(point);
        var n: u64 = 1024;
        var ctx: ChaseCtx = .{ .ring = self.l1, .start = &self.l1_at };
        var ns = try measure(1, ChaseCtx, &ctx, chaseWork, &n, self.quick);
        const l1_ns = ns / @as(f64, @floatFromInt(n));
        try report.write(out, .{ .device = "cpu", .probe = "check", .variant = "l1-latency", .x = x, .value = l1_ns, .unit = "ns", .ns = ns });

        n = 1024;
        ctx = .{ .ring = self.ram, .start = &self.ram_at };
        ns = try measure(1, ChaseCtx, &ctx, chaseWork, &n, self.quick);
        const ram_ns = ns / @as(f64, @floatFromInt(n));
        try report.write(out, .{ .device = "cpu", .probe = "check", .variant = "ram-latency", .x = x, .value = ram_ns, .unit = "ns", .ns = ns });

        const threads = @max(1, cores());
        const alu: AluCtx = .{ .f = aluFn(16, 16) };
        n = 1024;
        ns = try measure(threads, AluCtx, &alu, aluWork, &n, self.quick);
        const flops = 2.0 * 16.0 * 16.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(threads));
        const tflops = flops / ns / 1e3;
        try report.write(out, .{ .device = "cpu", .probe = "check", .variant = "fma-per-core", .x = x, .value = tflops, .unit = "TFLOP/s", .ns = ns });

        // The clock with one core busy, beside the markers: a marker that moved with the clock
        // is the boost, not the code. Not a marker itself (it does not set `disturbed`).
        if (try clockGhz(1, self.quick)) |ghz|
            try report.write(out, .{ .device = "cpu", .probe = "check", .variant = "clock", .x = x, .value = ghz, .unit = "GHz", .ns = 0 });

        try self.marks.append(gpa, .{ .l1_ns = l1_ns, .ram_ns = ram_ns, .fma_tflops = tflops });
    }

    /// Write `cpu,check,disturbed,0` (1 = some marker moved past the tolerance, 0 = none) and
    /// `cpu,check,worst-move` (the largest move, as a fraction), and say so on stderr.
    pub fn finish(self: *const Check, out: *std.Io.Writer) !bool {
        const worst = worstMove(self.marks.items);
        const disturbed = worst > disturbed_tolerance;
        try report.write(out, .{ .device = "cpu", .probe = "check", .variant = "worst-move", .x = 0, .value = worst, .unit = "fraction", .ns = 0 });
        try report.write(out, .{ .device = "cpu", .probe = "check", .variant = "disturbed", .x = 0, .value = if (disturbed) 1 else 0, .unit = "flag", .ns = 0 });
        if (disturbed) report.note("WARNING: this run looks disturbed: a marker moved {d:.0}% between checkpoints (tolerance {d:.0}%); something else was using the machine\n", .{ worst * 100, disturbed_tolerance * 100 });
        return disturbed;
    }
};

/// The largest relative move of any marker between its best and worst checkpoint
/// (worst / best - 1); 0 with fewer than two checkpoints.
pub fn worstMove(marks: []const Marks) f64 {
    if (marks.len < 2) return 0;
    var worst: f64 = 0;
    inline for (.{ "l1_ns", "ram_ns", "fma_tflops" }) |f| {
        var lo: f64 = std.math.inf(f64);
        var hi: f64 = 0;
        for (marks) |m| {
            lo = @min(lo, @field(m, f));
            hi = @max(hi, @field(m, f));
        }
        // A zero or negative marker would mean a broken timer; count it as fully disturbed
        // rather than dividing by it.
        const move = if (lo > 0) hi / lo - 1 else std.math.inf(f64);
        worst = @max(worst, move);
    }
    return worst;
}

/// Physical cores (one CPU per core in `order`), from the topology; all CPUs if unknown.
fn cores() u32 {
    return if (core_count != 0) core_count else cpus;
}

pub const all = [_][]const u8{ "clock", "alu", "chase", "stream", "branch", "sync", "alloc" };

pub fn run(gpa: std.mem.Allocator, out: *std.Io.Writer, name: []const u8, quick: bool) !bool {
    if (std.mem.eql(u8, name, "clock")) try probeClock(out, quick) //
    else if (std.mem.eql(u8, name, "alu")) try probeAlu(out, quick) //
    else if (std.mem.eql(u8, name, "chase")) try probeChase(gpa, out, quick) //
    else if (std.mem.eql(u8, name, "stream")) try probeStream(gpa, out, quick) //
    else if (std.mem.eql(u8, name, "branch")) try probeBranch(gpa, out, quick) //
    else if (std.mem.eql(u8, name, "sync")) try probeSync(gpa, out, quick) //
    else if (std.mem.eql(u8, name, "alloc")) try probeAlloc(out, quick) //
    else return false;
    return true;
}

test "cpu ring visits every line once" {
    var ring: [16 * 50]u32 = @splat(0);
    var rng = std.Random.DefaultPrng.init(1);
    try buildRing(std.testing.allocator, &ring, rng.random(), line_u32);
    var seen: [50]bool = @splat(false);
    var p: u32 = 0;
    for (0..50) |_| {
        try std.testing.expect(!seen[p / 16]);
        seen[p / 16] = true;
        p = ring[p];
    }
    try std.testing.expectEqual(@as(u32, 0), p);
}

test "cpu add chain runs every add" {
    const sum = addChain(1000) orelse return error.SkipZigTest;
    try std.testing.expectEqual(@as(u64, 1000 * clock_adds), sum);
    try std.testing.expectEqual(@as(?u64, 0), addChain(0));
}

test "cpu clock reads a plausible frequency" {
    // Full calibration (~0.3 s): the quick mode's few microseconds are mostly thread start-up.
    const ghz = try clockGhz(1, false) orelse return error.SkipZigTest;
    try std.testing.expect(ghz > 0.4 and ghz < 8);
}

test "cpu fma chains compute a*x+b" {
    // One round on chain i: 0.999 * (seed + i/1000) + 0.001, summed over chains and lanes.
    const got = fmaChains(4, 2, 1, 1.0);
    var want: f32 = 0;
    for (0..2) |i| want += 4 * (0.999 * (1.0 + @as(f32, @floatFromInt(i)) * 1e-3) + 0.001);
    try std.testing.expectApproxEqRel(want, got, 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 0.999 * 2.0 + 0.001), fmaChains(1, 1, 1, 2.0), 1e-6);
}

fn testProbe(name: []const u8) !void {
    report.progress = false;
    var sink: std.Io.Writer.Discarding = .init(&.{});
    try std.testing.expect(try run(std.testing.allocator, &sink.writer, name, true));
}

test "cpu probe alu runs (small sizes)" {
    try testProbe("alu");
}
test "cpu probe chase runs (small sizes)" {
    try testProbe("chase");
}
test "cpu probe stream runs (small sizes)" {
    try testProbe("stream");
}
test "cpu probe branch runs (small sizes)" {
    try testProbe("branch");
}
test "cpu probe sync runs (small sizes)" {
    try testProbe("sync");
}
test "cpu probe alloc runs (small sizes)" {
    try testProbe("alloc");
}

test "cpu check: worstMove is the largest best-to-worst ratio of any marker" {
    const m = Marks{ .l1_ns = 1, .ram_ns = 90, .fma_tflops = 2.5 };
    try std.testing.expectEqual(@as(f64, 0), worstMove(&.{}));
    try std.testing.expectEqual(@as(f64, 0), worstMove(&.{m}));
    try std.testing.expectEqual(@as(f64, 0), worstMove(&.{ m, m, m }));
    // L1 latency 3x at one checkpoint (what a disturbed run showed): a move of 2.0.
    var slow = m;
    slow.l1_ns = 3;
    try std.testing.expectApproxEqRel(@as(f64, 2.0), worstMove(&.{ m, slow, m }), 1e-12);
    // FMA throughput dropping 20% counts as a 25% move (best / worst - 1), same as latency.
    var weak = m;
    weak.fma_tflops = 2.0;
    try std.testing.expectApproxEqRel(@as(f64, 0.25), worstMove(&.{ weak, m }), 1e-12);
    // RAM latency within the tolerance: not flagged by itself.
    var ram = m;
    ram.ram_ns = 95;
    try std.testing.expect(worstMove(&.{ m, ram }) < disturbed_tolerance);
    // A zero marker (broken timer) counts as fully disturbed instead of dividing by zero.
    var zero = m;
    zero.ram_ns = 0;
    try std.testing.expect(std.math.isInf(worstMove(&.{ m, zero })));
}

test "cpu check: takes markers at each checkpoint and writes the flag rows" {
    report.progress = false;
    var check = try Check.init(std.testing.allocator, true);
    defer check.deinit(std.testing.allocator);
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try check.take(std.testing.allocator, &w, 0);
    try check.take(std.testing.allocator, &w, 1);
    try std.testing.expectEqual(@as(usize, 2), check.marks.items.len);
    for (check.marks.items) |mk| {
        try std.testing.expect(mk.l1_ns > 0 and mk.ram_ns > 0 and mk.fma_tflops > 0);
    }
    // Debug timings are noisy, so the flag may go either way here; it must match the move.
    const disturbed = try check.finish(&w);
    try std.testing.expectEqual(worstMove(check.marks.items) > disturbed_tolerance, disturbed);
    const text = w.buffered();
    const flag_row = if (disturbed) "cpu,check,disturbed,0,0,1.0000," else "cpu,check,disturbed,0,0,0.0000,";
    inline for (.{ "cpu,check,l1-latency,0,", "cpu,check,ram-latency,1,", "cpu,check,fma-per-core,1,", "cpu,check,worst-move,0,0," }) |want| {
        try std.testing.expect(std.mem.find(u8, text, want) != null);
    }
    try std.testing.expect(std.mem.find(u8, text, flag_row) != null);
}
