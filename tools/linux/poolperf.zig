// Experiment E4 (2026-11-27): the pool round trip (workers awake) on Linux reads ~120 or ~150-160
// ns per round trip and switches between the two within one process (poolexp.zig). This counts,
// per batch of 20000 round trips and per thread, cycles, instructions, cache lines filled from
// another core's cache (ls_any_fills_from_sys.local_ccx, raw 0x244), from the own L2 (0x144),
// and pipeline resyncs (bp_redirects.resync, 0x19f), so fast and slow batches can be compared.
// Each thread opens its own counters (pid 0, any CPU): allowed at perf_event_paranoid 1.
// Build: zig build-exe -OReleaseFast --dep zstrata -Mroot=tools/linux/poolperf.zig -Mzstrata=src/lib.zig
// zsnag:allow R010 an experiment that prints to the terminal
const std = @import("std");
const zs = @import("zstrata");
const linux = std.os.linux;

const n_ev = 5;
const names = [n_ev][]const u8{ "cycles", "instr", "ccx", "l2", "resync" };

fn openAll() [n_ev]i32 {
    const evs = [n_ev]struct { t: linux.PERF.TYPE, c: u64 }{
        .{ .t = .HARDWARE, .c = @backingInt(linux.PERF.COUNT.HW.CPU_CYCLES) },
        .{ .t = .HARDWARE, .c = @backingInt(linux.PERF.COUNT.HW.INSTRUCTIONS) },
        .{ .t = .RAW, .c = 0x244 },
        .{ .t = .RAW, .c = 0x144 },
        .{ .t = .RAW, .c = 0x19f },
    };
    var fds: [n_ev]i32 = undefined;
    for (evs, 0..) |e, i| {
        var attr: linux.perf_event_attr = .{ .type = e.t, .config = e.c };
        attr.flags.exclude_kernel = true;
        attr.flags.exclude_hv = true;
        const r = linux.perf_event_open(&attr, 0, -1, -1, 0);
        if (linux.errno(r) != .SUCCESS) {
            std.debug.print("perf_event_open {s}: {t}\n", .{ names[i], linux.errno(r) });
            std.process.exit(1);
        }
        fds[i] = @intCast(r); // a successful syscall returns a small fd
    }
    return fds;
}

fn readAll(fds: [n_ev]i32) [n_ev]u64 {
    var v: [n_ev]u64 = undefined;
    for (fds, 0..) |fd, i| {
        var x: u64 = 0;
        if (linux.read(fd, @ptrCast(&x), 8) != 8) x = 0;
        v[i] = x;
    }
    return v;
}

var helper_fds: ?[n_ev]i32 = null;
var helper_snap: [n_ev]u64 = @splat(0);
var want_snap = false;
fn task(_: *anyopaque, worker: usize, _: usize, _: usize) void {
    if (worker == 0) return;
    if (helper_fds == null) helper_fds = openAll();
    if (want_snap) {
        helper_snap = readAll(helper_fds.?);
        want_snap = false;
    }
}
fn empty(_: *anyopaque, _: usize, _: usize, _: usize) void {}

fn nowNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Asks the helper for a snapshot of its counters and waits until it has taken one.
fn helperSnap(pool: *zs.Pool) [n_ev]u64 {
    var dummy: u8 = 0;
    want_snap = true;
    while (want_snap) pool.parallelFor(64, &dummy, task, 1);
    return helper_snap;
}

pub fn main(init: std.process.Init) !void {
    const pool = try zs.Pool.init(init.gpa, 2);
    defer pool.deinit();
    const mine = openAll();
    _ = helperSnap(pool);
    const batch = 20000;
    var dummy: u8 = 0;
    std.debug.print("ns/rt  | main: cycles instr ccx l2 resync per rt | helper: same\n", .{});
    for (0..60) |_| {
        const m0 = readAll(mine);
        const h0 = helperSnap(pool);
        const t0 = nowNs();
        for (0..batch) |_| pool.parallelFor(8, &dummy, empty, 1);
        const ns = @as(f64, @floatFromInt(nowNs() - t0)) / batch;
        const m1 = readAll(mine);
        const h1 = helperSnap(pool);
        std.debug.print("{d:6.1} |", .{ns});
        for (0..n_ev) |i| std.debug.print(" {d:7.2}", .{@as(f64, @floatFromInt(m1[i] - m0[i])) / batch});
        std.debug.print(" |", .{});
        // The helper's window also covers two snapshot round trips: negligible against 20000.
        for (0..n_ev) |i| std.debug.print(" {d:7.2}", .{@as(f64, @floatFromInt(h1[i] - h0[i])) / batch});
        std.debug.print("\n", .{});
    }
}
