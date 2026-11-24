// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! One CSV schema for every probe on every device, so the page can put CPU, link and GPU
//! rows side by side. GPU rows also carry the driver's compile statistics; CPU rows leave
//! those columns empty.

const std = @import("std");

/// What the GPU driver says it compiled (null on the CPU, or when the driver has no such
/// statistic): registers and spills per lane, scratch memory, shared memory per workgroup,
/// subgroups that fit on one SIMD, instructions paired for dual issue, scalar loads.
pub const Stats = struct {
    vgprs: ?u64 = null,
    sgprs: ?u64 = null,
    spilled_vgprs: ?u64 = null,
    scratch: ?u64 = null,
    lds: ?u64 = null,
    waves: ?u64 = null,
    vopd: ?u64 = null,
    smem: ?u64 = null,
};

pub const Row = struct {
    /// Which device the row measures: "cpu", "link" or "gpu".
    device: []const u8,
    probe: []const u8,
    variant: []const u8,
    x: f64,
    y: f64 = 0,
    value: f64,
    unit: []const u8,
    ns: f64,
    stats: Stats = .{},
};

pub const header = "device,probe,variant,x,y,value,unit,ms,vgprs,sgprs,spilled_vgprs,scratch_bytes,lds_bytes,subgroups_per_simd,vopd,smem\n";

/// Progress lines on stderr. Off in tests: the test runner counts any stderr output as a
/// failure.
pub var progress = true;

pub fn note(comptime fmt: []const u8, args: anytype) void {
    if (progress) std.debug.print(fmt, args);
}

pub fn write(out: *std.Io.Writer, r: Row) !void {
    try out.print("{s},{s},{s},{d},{d},{d:.4},{s},{d:.4}", .{ r.device, r.probe, r.variant, r.x, r.y, r.value, r.unit, r.ns / 1e6 });
    inline for (.{ "vgprs", "sgprs", "spilled_vgprs", "scratch", "lds", "waves", "vopd", "smem" }) |f| {
        if (@field(r.stats, f)) |v| try out.print(",{d}", .{v}) else try out.writeAll(",");
    }
    try out.writeAll("\n");
    try out.flush();
    note("{s:<4} {s:<9} {s:<18} x={d:<11} y={d:<6} {d:>12.3} {s}\n", .{ r.device, r.probe, r.variant, r.x, r.y, r.value, r.unit });
}

/// Median of a small sample; sorts `xs` in place.
pub fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}
