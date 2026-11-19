// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! zstrata's own parallel-for pool: the thing the sync probe times and the crossovers use for
//! "all threads". It is kept here, not borrowed from a project being measured, so the benchmark
//! has no dependency on what it is meant to check. The design (it started as zarbor's pool):
//! workers spin on an epoch counter (no syscall on the fast path), and past `spin_budget` sleep
//! on it, a futex on Linux and WaitOnAddress on Windows (yield elsewhere). The submitter takes
//! part, then waits for the barrier by spinning, then yielding. A shared cursor, not a static
//! partition, so a descheduled worker cannot hold the barrier.

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;

/// Zig has no portable futex: the raw Linux syscall, or ntdll's RtlWaitOnAddress (behind
/// kernel32's WaitOnAddress) on Windows; elsewhere a yield loop, correct but busy.
pub const can_park = builtin.os.tag == .linux or builtin.os.tag == .windows;

const wait_op: linux.FUTEX_OP = .{ .cmd = .WAIT, .private = true };
const wake_op: linux.FUTEX_OP = .{ .cmd = .WAKE, .private = true };

/// Sleeps until `v` != `expect`; returns at once if already so (closes the publish race).
fn park(v: *std.atomic.Value(u32), expect: u32) void {
    if (builtin.os.tag == .windows) {
        // Same contract: compares *v with `expect` and sleeps only while they are equal.
        _ = std.os.windows.ntdll.RtlWaitOnAddress(&v.raw, &expect, @sizeOf(u32), null);
        return;
    }
    _ = linux.futex_4arg(&v.raw, wait_op, expect, null);
}

fn unparkAll(v: *std.atomic.Value(u32)) void {
    if (builtin.os.tag == .windows) return std.os.windows.ntdll.RtlWakeAddressAll(&v.raw);
    _ = linux.futex_3arg(&v.raw, wake_op, std.math.maxInt(i32));
}

const cache_line = std.atomic.cache_line;

/// Spins before a worker sleeps, or before the submitter starts yielding.
pub const spin_budget: u32 = 8192;

/// Worker stacks: the tasks here are tiny. Zig's default (16 MiB) is committed up front on
/// Windows, which costs time at spawn and commit charge for every helper.
const worker_stack = 1 << 20;

/// Half-open range plus worker id.
pub const TaskFn = *const fn (ctx: *anyopaque, worker: usize, begin: usize, end: usize) void;

pub const Pool = struct {
    gpa: std.mem.Allocator,
    /// Spawned helpers; the submitter is worker 0, so concurrency is `threads.len + 1`.
    threads: []std.Thread,

    // Job description; published by the release store to `epoch`.
    task: TaskFn = undefined,
    ctx: *anyopaque = undefined,
    total: usize = 0,
    chunk: usize = 1,

    // Written by every worker: separate cache lines.
    cursor: std.atomic.Value(usize) align(cache_line) = .init(0),
    done: std.atomic.Value(usize) align(cache_line) = .init(0),
    epoch: std.atomic.Value(u32) align(cache_line) = .init(0),
    quit: std.atomic.Value(bool) align(cache_line) = .init(false),
    /// Workers asleep on `epoch`; the publisher skips the wake call when none are.
    parked: std.atomic.Value(u32) align(cache_line) = .init(0),

    /// `n_threads` 0 = one per logical core. Heap-allocated: workers hold a `*Pool` for life,
    /// so its address must be stable. Caller frees with `deinit`.
    pub fn init(gpa: std.mem.Allocator, n_threads: u32) !*Pool {
        const cores = std.Thread.getCpuCount() catch 1;
        const want = if (n_threads == 0) cores else n_threads;
        const helpers = if (want <= 1) 0 else want - 1;

        const p = try gpa.create(Pool);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .threads = try gpa.alloc(std.Thread, helpers) };
        errdefer gpa.free(p.threads);

        // The pool is fully initialised before the first worker reads it.
        var spawned: usize = 0;
        errdefer {
            p.quit.store(true, .release);
            _ = p.epoch.fetchAdd(1, .release);
            if (can_park) unparkAll(&p.epoch);
            for (p.threads[0..spawned]) |t| t.join();
        }
        while (spawned < helpers) : (spawned += 1) {
            p.threads[spawned] = try std.Thread.spawn(.{ .stack_size = worker_stack }, workerMain, .{ p, spawned + 1 });
        }
        return p;
    }

    pub fn deinit(p: *Pool) void {
        const gpa = p.gpa;
        const threads = p.threads;
        p.quit.store(true, .release);
        // Bump the epoch so sleeping workers wake and see `quit`.
        _ = p.epoch.fetchAdd(1, .release);
        if (can_park) unparkAll(&p.epoch);
        for (threads) |t| t.join();
        gpa.free(threads);
        gpa.destroy(p);
    }

    pub fn workerCount(p: *const Pool) usize {
        return p.threads.len + 1;
    }

    /// The size of the pieces `parallelFor(total, .., min_chunk)` hands out, or `total` when it
    /// would run the whole range in one call. Every piece begins at a multiple of this and all
    /// but the last have this length, so a task can use `begin / chunkFor(..)` as the piece's
    /// index (for per-piece scratch or results reduced in piece order), and
    /// `(total + chunk - 1) / chunk` is how many pieces there are. Zero for `total == 0`.
    pub fn chunkFor(p: *const Pool, total: usize, min_chunk: usize) usize {
        const workers = p.workerCount();
        if (workers == 1 or total <= min_chunk) return total;
        // Several chunks per worker so a slow core cannot hold the barrier; few enough that the
        // cursor is not the bottleneck.
        const chunk = (total + workers * 4 - 1) / (workers * 4);
        return @max(chunk, min_chunk);
    }

    /// Run `task` over `[0, total)`. At or under `min_chunk` it runs inline (the barrier would
    /// cost more). The pieces are the ones `chunkFor` describes.
    pub fn parallelFor(p: *Pool, total: usize, ctx: *anyopaque, task: TaskFn, min_chunk: usize) void {
        if (total == 0) return;
        const workers = p.workerCount();
        if (workers == 1 or total <= min_chunk) {
            task(ctx, 0, 0, total);
            return;
        }
        const chunk = p.chunkFor(total, min_chunk);

        p.task = task;
        p.ctx = ctx;
        p.total = total;
        p.chunk = chunk;
        p.cursor.store(0, .monotonic);
        p.done.store(0, .monotonic);

        // Release: visible to any worker acquiring the new epoch.
        _ = p.epoch.fetchAdd(1, .release);
        if (can_park and p.parked.load(.acquire) != 0) unparkAll(&p.epoch);

        drain(p, 0);

        var spins: u32 = 0;
        while (p.done.load(.acquire) < p.threads.len) {
            spins +%= 1;
            if (spins < spin_budget) std.atomic.spinLoopHint() else std.Thread.yield() catch {}; // zsnag:ok — a failed yield just means spin again
        }
    }
};

fn drain(p: *Pool, worker: usize) void {
    while (true) {
        const begin = p.cursor.fetchAdd(p.chunk, .monotonic);
        if (begin >= p.total) break;
        const end = @min(begin + p.chunk, p.total);
        p.task(p.ctx, worker, begin, end);
    }
}

fn workerMain(p: *Pool, id: usize) void {
    var seen: u32 = 0;
    while (true) {
        var spins: u32 = 0;
        while (true) {
            if (p.quit.load(.acquire)) return;
            const e = p.epoch.load(.acquire);
            if (e != seen) {
                seen = e;
                break;
            }
            spins +%= 1;
            if (spins < spin_budget) {
                std.atomic.spinLoopHint();
                continue;
            }
            if (!can_park) {
                std.Thread.yield() catch {}; // zsnag:ok — a failed yield just means spin again
                continue;
            }
            // Past the spin budget: sleep. Announce before re-reading the epoch: a later bump
            // sees `parked` and wakes us; an earlier one fails the compare in `park`.
            _ = p.parked.fetchAdd(1, .acq_rel);
            if (p.epoch.load(.acquire) == e and !p.quit.load(.acquire)) park(&p.epoch, e);
            _ = p.parked.fetchSub(1, .acq_rel);
            spins = 0;
        }
        if (p.quit.load(.acquire)) return;
        drain(p, id);
        _ = p.done.fetchAdd(1, .release);
    }
}

// ---------------------------------------------------------------- tests

const testing = std.testing;

const Count = struct {
    vals: []u32,
    fn run(ctx: *anyopaque, _: usize, begin: usize, end: usize) void {
        const self: *Count = @ptrCast(@alignCast(ctx));
        for (self.vals[begin..end]) |*v| v.* += 1;
    }
};

test "cpu pool covers every index exactly once, round after round (1, 3, 16 threads)" {
    const gpa = testing.allocator;
    for ([_]u32{ 1, 3, 16 }) |threads| {
        const pool = try Pool.init(gpa, threads);
        defer pool.deinit();
        const vals = try gpa.alloc(u32, 100_003);
        defer gpa.free(vals);
        @memset(vals, 0);
        var ctx: Count = .{ .vals = vals };
        for (0..50) |_| pool.parallelFor(vals.len, &ctx, Count.run, 1);
        for (vals) |v| try testing.expectEqual(@as(u32, 50), v);
    }
}

const Pieces = struct {
    chunk: usize,
    total: usize,
    /// How many times each piece index was handed out.
    seen: []std.atomic.Value(u32),
    bad: std.atomic.Value(u32) = .init(0),
    fn run(ctx: *anyopaque, _: usize, begin: usize, end: usize) void {
        const self: *Pieces = @ptrCast(@alignCast(ctx));
        const full = end - begin == self.chunk;
        const last = end == self.total and end - begin <= self.chunk;
        if (begin % self.chunk != 0 or !(full or last)) _ = self.bad.fetchAdd(1, .monotonic);
        _ = self.seen[begin / self.chunk].fetchAdd(1, .monotonic);
    }
};

test "cpu pool chunkFor describes the pieces parallelFor hands out (1, 3, 16 threads)" {
    const gpa = testing.allocator;
    for ([_]u32{ 1, 3, 16 }) |threads| {
        const pool = try Pool.init(gpa, threads);
        defer pool.deinit();
        try testing.expectEqual(@as(usize, 0), pool.chunkFor(0, 1));
        for ([_]usize{ 1, 7, 64, 65, 1000, 100_003 }) |total| {
            for ([_]usize{ 1, 7, 64, 5000 }) |min_chunk| {
                const chunk = pool.chunkFor(total, min_chunk);
                const pieces = (total + chunk - 1) / chunk;
                const seen = try gpa.alloc(std.atomic.Value(u32), pieces);
                defer gpa.free(seen);
                for (seen) |*s| s.* = .init(0);
                var ctx: Pieces = .{ .chunk = chunk, .total = total, .seen = seen };
                pool.parallelFor(total, &ctx, Pieces.run, min_chunk);
                try testing.expectEqual(@as(u32, 0), ctx.bad.load(.monotonic));
                for (seen) |*s| try testing.expectEqual(@as(u32, 1), s.load(.monotonic));
            }
        }
    }
}

test "cpu pool helpers really sleep when idle and wake for the next round (1, 3, 16 threads)" {
    // Asserts the mechanism, not only the result: after a gap every helper is counted in
    // `parked`, and the next round still covers every index exactly once (a lost wake-up would
    // hang here).
    const gpa = testing.allocator;
    for ([_]u32{ 1, 3, 16 }) |threads| {
        const pool = try Pool.init(gpa, threads);
        defer pool.deinit();
        const vals = try gpa.alloc(u32, 50_000);
        defer gpa.free(vals);
        @memset(vals, 0);
        var ctx: Count = .{ .vals = vals };
        const rounds = 4;
        for (0..rounds) |_| {
            pool.parallelFor(vals.len, &ctx, Count.run, 7);
            if (!can_park or pool.threads.len == 0) continue;
            var waited: u32 = 0; // bounded: 2 s
            while (pool.parked.load(.acquire) < pool.threads.len) : (waited += 1) {
                if (waited >= 2000) return error.HelpersNeverParked;
                try testing.io.sleep(.fromMilliseconds(1), .awake);
            }
        }
        for (vals) |v| try testing.expectEqual(@as(u32, rounds), v);
    }
}
