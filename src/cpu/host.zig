// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! The host-OS layer of the probes: a monotonic clock, pinning the calling thread to one CPU,
//! a short sleep, and the core topology. Linux keeps the calls it always used; Windows uses its
//! own (QueryPerformanceCounter, SetThreadAffinityMask, Sleep, GetLogicalProcessorInformationEx).

const std = @import("std");
const builtin = @import("builtin");
const report = @import("../report.zig");

const is_windows = builtin.os.tag == .windows;

const simd = @import("../lib/simd.zig");

/// See lib/simd.zig: fused multiply-add only where the build's CPU has the instruction.
pub const fused_is_an_instruction = simd.fused_is_an_instruction;
pub const mulAdd = simd.mulAdd;

/// Monotonic nanoseconds. Windows: QueryPerformanceCounter, whose tick is reported by
/// QueryPerformanceFrequency (so its resolution can be coarser than 1 ns).
pub fn nowNs() u64 {
    if (is_windows) {
        var ticks: i64 = 0;
        _ = std.os.windows.ntdll.RtlQueryPerformanceCounter(&ticks);
        // QPC is documented never to go negative; clamp rather than trust a cast.
        return cpuDomainNs(@intCast(@max(ticks, 0)));
    }
    const linux = std.os.linux;
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// A raw reading of the clock `nowNs` uses (QPC ticks on Windows, ns on Linux), in the same ns
/// `nowNs` returns. GPU calibration hands back raw readings of this clock.
pub fn cpuDomainNs(raw: u64) u64 {
    if (!is_windows) return raw;
    const t: u128 = raw;
    // ns = ticks * 1e9 / freq; u128 so the product cannot overflow, and the result fits u64
    // for ~584 years of uptime.
    return @truncate(t * std.time.ns_per_s / qpcFrequency());
}

/// CPU time the calling thread has used, in ns. Linux: CLOCK_THREAD_CPUTIME_ID. Windows:
/// QueryThreadCycleTime (cycles the thread actually ran, counted at the TSC rate) converted with
/// a rate calibrated once against QPC. GetThreadTimes would be simpler but is charged in whole
/// scheduler ticks (15.6 ms), far too coarse for a 50 us poll.
pub fn threadCpuNs() u64 {
    if (is_windows) {
        var cycles: u64 = 0;
        if (win.QueryThreadCycleTime(std.os.windows.GetCurrentThread(), &cycles) == 0) return 0;
        return @intFromFloat(@as(f64, @floatFromInt(cycles)) * nsPerCycle());
    }
    const linux = std.os.linux;
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.THREAD_CPUTIME_ID, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// ns per cycle of QueryThreadCycleTime: spin ~10 ms of QPC time and count this thread's cycles
/// (a busy thread runs all of it). Read once; the TSC rate is fixed.
var ns_per_cycle = std.atomic.Value(u64).init(0); // f64 bits; 0 = not yet measured
fn nsPerCycle() f64 {
    const cached = ns_per_cycle.load(.monotonic);
    if (cached != 0) return @bitCast(cached);
    const h = std.os.windows.GetCurrentThread();
    var c0: u64 = 0;
    var c1: u64 = 0;
    _ = win.QueryThreadCycleTime(h, &c0);
    const t0 = nowNs();
    while (nowNs() - t0 < 10 * std.time.ns_per_ms) std.atomic.spinLoopHint();
    const t1 = nowNs();
    _ = win.QueryThreadCycleTime(h, &c1);
    // A preempted spin counts fewer cycles and slightly overstates ns/cycle; 10 ms on an idle
    // core keeps that small. Guard the divide: no cycles counted means no usable rate.
    const r: f64 = if (c1 > c0) @as(f64, @floatFromInt(t1 - t0)) / @as(f64, @floatFromInt(c1 - c0)) else 0;
    if (r > 0) ns_per_cycle.store(@bitCast(r), .monotonic);
    return r;
}

/// QueryPerformanceFrequency, read once; it is fixed at boot.
var qpc_freq = std.atomic.Value(u64).init(0);
fn qpcFrequency() u64 {
    const cached = qpc_freq.load(.monotonic);
    if (cached != 0) return cached;
    var f: i64 = 0;
    _ = std.os.windows.ntdll.RtlQueryPerformanceFrequency(&f);
    const freq: u64 = if (f > 0) @intCast(f) else std.time.ns_per_s; // documented never to fail since XP
    qpc_freq.store(freq, .monotonic);
    return freq;
}

/// Pin the calling thread to `cpu`. A failure is noted, not fatal: the row is then unpinned.
pub fn pin(cpu: u32) void {
    if (is_windows) {
        if (cpu >= @bitSizeOf(usize)) return report.note("could not pin to cpu {d}: past one processor group\n", .{cpu});
        const mask = @as(usize, 1) << @intCast(cpu);
        if (win.SetThreadAffinityMask(std.os.windows.GetCurrentThread(), mask) == 0)
            report.note("could not pin to cpu {d}: SetThreadAffinityMask failed\n", .{cpu});
        return;
    }
    const linux = std.os.linux;
    var set: linux.cpu_set_t = @splat(0);
    set[cpu / @bitSizeOf(usize)] |= @as(usize, 1) << @intCast(cpu % @bitSizeOf(usize));
    linux.sched_setaffinity(0, &set) catch |e| report.note("could not pin to cpu {d}: {t}\n", .{ cpu, e });
}

/// Sleep about `ms` milliseconds. Windows rounds up to its timer tick (often 15.6 ms), which only
/// lengthens the idle gap it is used for.
pub fn sleepMs(ms: u32) void {
    if (is_windows) return win.Sleep(ms);
    const linux = std.os.linux;
    const req: linux.timespec = .{ .sec = @intCast(ms / 1000), .nsec = @intCast(@as(u64, ms % 1000) * 1_000_000) };
    _ = linux.nanosleep(&req, null);
}

/// The CPUs, grouped by core: `firsts` gets one CPU per core, `seconds` every other thread of
/// each core (the SMT partners). Returns false when the topology cannot be read.
pub fn topology(io: std.Io, firsts: *[64]u32, nf: *u32, seconds: *[64]u32, ns: *u32) bool {
    nf.* = 0;
    ns.* = 0;
    if (is_windows) return windowsTopology(firsts, nf, seconds, ns);
    var seen: [64]bool = @splat(false);
    var cpu: u32 = 0;
    while (cpu < 64) : (cpu += 1) {
        var path_buf: [96]u8 = undefined;
        const path = std.mem.print(&path_buf, "/sys/devices/system/cpu/cpu{d}/topology/thread_siblings_list", .{cpu}) catch break;
        var buf: [32]u8 = undefined;
        const text = std.Io.Dir.cwd().readFile(io, path, &buf) catch break;
        if (seen[cpu]) continue;
        var it = std.mem.tokenizeAny(u8, text, ",-\n");
        var first = true;
        while (it.next()) |tok| {
            const c = std.fmt.parseInt(u32, tok, 10) catch continue;
            if (c >= 64) continue;
            seen[c] = true;
            if (first) {
                firsts[nf.*] = c;
                nf.* += 1;
            } else {
                seconds[ns.*] = c;
                ns.* += 1;
            }
            first = false;
        }
    }
    return nf.* != 0;
}

/// One `RelationProcessorCore` record per core from GetLogicalProcessorInformationEx; its
/// group-0 affinity mask lists the core's logical CPUs (lowest first, then its SMT partners).
fn windowsTopology(firsts: *[64]u32, nf: *u32, seconds: *[64]u32, ns: *u32) bool {
    var buf: [16 * 1024]u8 align(8) = undefined;
    var len: u32 = buf.len;
    if (win.GetLogicalProcessorInformationEx(win.relation_processor_core, &buf, &len) == 0) return false;
    // Record: Relationship u32, Size u32, then PROCESSOR_RELATIONSHIP { Flags u8,
    // EfficiencyClass u8, Reserved [20]u8, GroupCount u16, GroupMask [GroupCount]GROUP_AFFINITY }
    // where GROUP_AFFINITY is { Mask KAFFINITY, Group u16, Reserved [3]u16 } at 8-byte alignment.
    var off: usize = 0;
    while (off + 48 <= len) {
        const size = std.mem.readInt(u32, buf[off + 4 ..][0..4], .little);
        if (size < 48 or off + size > len) return false;
        const rel = std.mem.readInt(u32, buf[off..][0..4], .little);
        const group_count = std.mem.readInt(u16, buf[off + 30 ..][0..2], .little);
        if (rel == win.relation_processor_core and group_count >= 1) {
            const mask = std.mem.readInt(u64, buf[off + 32 ..][0..8], .little);
            const group = std.mem.readInt(u16, buf[off + 40 ..][0..2], .little);
            if (group == 0) {
                var m = mask;
                var first = true;
                while (m != 0) : (m &= m - 1) {
                    const c: u32 = @ctz(m);
                    if (first) {
                        firsts[nf.*] = c;
                        nf.* += 1;
                    } else {
                        seconds[ns.*] = c;
                        ns.* += 1;
                    }
                    first = false;
                }
            }
        }
        off += size;
    }
    return nf.* != 0;
}

/// Ask Windows not to throttle this process: opt out of execution-speed throttling (EcoQoS)
/// and of timer requests being ignored, the "HighQoS" setting of SetProcessInformation. Needs
/// no privilege. Windows may give lower-priority processes (these runs are at BelowNormal) a
/// lower quality of service otherwise. Returns false off Windows or when the call fails.
pub fn noThrottling() bool {
    if (!is_windows) return false;
    var state: win.PowerThrottlingState = .{
        .version = win.power_throttling_current_version,
        .control_mask = win.power_throttling_execution_speed | win.power_throttling_ignore_timer_resolution,
        .state_mask = 0, // 0 = off for every mechanism named in control_mask
    };
    return win.SetProcessInformation(std.os.windows.GetCurrentProcess(), win.process_power_throttling, &state, @sizeOf(win.PowerThrottlingState)) != 0;
}

const win = struct {
    const relation_processor_core: u32 = 0;
    const process_power_throttling: c_int = 4; // PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
    const power_throttling_current_version: u32 = 1;
    const power_throttling_execution_speed: u32 = 0x1;
    const power_throttling_ignore_timer_resolution: u32 = 0x4;
    const PowerThrottlingState = extern struct { version: u32, control_mask: u32, state_mask: u32 };
    extern "kernel32" fn SetProcessInformation(process: std.os.windows.HANDLE, class: c_int, info: *anyopaque, size: u32) callconv(.winapi) c_int;
    extern "kernel32" fn SetThreadAffinityMask(thread: std.os.windows.HANDLE, mask: usize) callconv(.winapi) usize;
    extern "kernel32" fn Sleep(ms: u32) callconv(.winapi) void;
    extern "kernel32" fn QueryThreadCycleTime(thread: std.os.windows.HANDLE, cycles: *u64) callconv(.winapi) c_int;
    extern "kernel32" fn GetLogicalProcessorInformationEx(relationship: u32, buffer: ?*anyopaque, length: *u32) callconv(.winapi) c_int;
};
