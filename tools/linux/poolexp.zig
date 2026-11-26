// Experiment (2026-11-26): why the pool round trip (workers awake) is bimodal on Linux.
// Prints us per round trip for 40 batches of 20000 round trips: unpinned, then submitter and
// helper pinned to two cores.
// zsnag:allow R010 an experiment that prints to the terminal
const std = @import("std");
const zs = @import("zstrata");
const linux = std.os.linux;

fn nowNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}
fn pin(cpu: usize) void {
    var set: linux.cpu_set_t = @splat(0);
    set[0] = @as(usize, 1) << @intCast(cpu);
    linux.sched_setaffinity(0, &set) catch {}; // zsnag:ok unpinned still measures
}
const adds = blk: {
    var r: []const u8 = "";
    for (0..32) |_| r = r ++ "addq $1, %[x]\n";
    break :blk r;
};

/// GHz of the calling core from 20M dependent one-cycle adds.
fn ghz() f64 {
    var x: u64 = 0;
    var left: u64 = 625_000;
    const t0 = nowNs();
    asm volatile ("1:\n" ++ adds ++ "decq %[left]\njnz 1b"
        : [x] "+r" (x),
          [left] "+r" (left),
        :
        : .{ .cc = true });
    return 20e6 / @as(f64, @floatFromInt(nowNs() - t0));
}

var helper_cpu: usize = 99;
fn task(_: *anyopaque, worker: usize, _: usize, _: usize) void {
    if (worker != 0 and helper_cpu != 99) {
        pin(helper_cpu);
        helper_cpu = 99;
    }
}
var helper_ghz: f64 = 0;
fn clockTask(_: *anyopaque, worker: usize, _: usize, _: usize) void {
    if (worker != 0) helper_ghz = ghz();
}
fn empty(_: *anyopaque, _: usize, _: usize, _: usize) void {}

fn series(pool: *zs.Pool, n: u32, label: []const u8) void {
    var dummy: u8 = 0;
    std.debug.print("{s:<22}", .{label});
    for (0..16) |_| {
        const t0 = nowNs();
        for (0..20000) |_| pool.parallelFor(n * 4, &dummy, empty, 1);
        const ns = @as(f64, @floatFromInt(nowNs() - t0)) / 20000.0;
        helper_ghz = 0;
        // Until the helper took a piece (the submitter may take them all).
        while (helper_ghz == 0) pool.parallelFor(64, &dummy, clockTask, 1);
        std.debug.print(" {d:.0}@{d:.2}", .{ ns, helper_ghz });
    }
    std.debug.print("  ns\n", .{});
}

pub fn main(init: std.process.Init) !void {
    const pool = try zs.Pool.init(init.gpa, 2);
    defer pool.deinit();
    var dummy: u8 = 0;
    series(pool, 2, "unpinned");
    for ([_][2]usize{ .{ 0, 1 }, .{ 2, 5 }, .{ 0, 8 }, .{ 3, 4 } }) |pair| {
        pin(pair[0]);
        helper_cpu = pair[1];
        for (0..2000) |_| pool.parallelFor(8, &dummy, task, 1);
        var buf: [32]u8 = undefined;
        series(pool, 2, std.mem.print(&buf, "pinned {d},{d}", .{ pair[0], pair[1] }) catch "pinned");
    }
}
