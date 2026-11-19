// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! How much each number moves between runs. Reads several zstrata CSVs, matches rows by
//! (device, probe, variant, x, y), and writes one row per measurement: runs seen, median, min,
//! max and spread = (max - min) / median. A summary goes to stderr: rows per spread band, per
//! device and probe, and the most variable rows.
//!
//!     zig build spread -- docs/data/runs/run*.csv > docs/data/strata-spread.csv

const std = @import("std");

const Entry = struct {
    unit: []const u8,
    values: std.ArrayList(f64) = .empty,
};

const Stat = struct {
    key: []const u8,
    unit: []const u8,
    runs: usize,
    median: f64,
    min: f64,
    max: f64,

    fn spread(s: Stat) f64 {
        return if (s.median == 0) 0 else (s.max - s.min) / @abs(s.median);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len < 3) {
        std.log.err("usage: spread run1.csv run2.csv [...]", .{});
        std.process.exit(2);
    }

    // Everything lives in the arena, freed when main returns.
    var rows: std.array_hash_map.String(Entry) = .empty;
    const runs = try arena.alloc(Run, argv.len - 1);
    for (argv[1..], runs) |path, *run| {
        run.* = .{ .path = path };
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(64 << 20));
        var lines = std.mem.tokenizeScalar(u8, text, '\n');
        _ = lines.next(); // header
        while (lines.next()) |line| {
            // device,probe,variant,x,y,value,unit,...
            var f: [7][]const u8 = undefined;
            var it = std.mem.splitScalar(u8, line, ',');
            for (&f) |*field| field.* = it.next() orelse return error.ShortRow;
            const key = try std.fmt.allocPrint(arena, "{s},{s},{s},{s},{s}", .{ f[0], f[1], f[2], f[3], f[4] });
            const gop = try rows.getOrPut(arena, key);
            if (!gop.found_existing) gop.value_ptr.* = .{ .unit = f[6] };
            const value = try std.fmt.parseFloat(f64, f[5]);
            try gop.value_ptr.values.append(arena, value);
            if (std.mem.eql(u8, f[0], "cpu") and std.mem.eql(u8, f[1], "check")) try run.add(arena, f[2], value);
        }
    }

    var stats: std.ArrayList(Stat) = .empty;
    for (rows.keys(), rows.values()) |key, e| {
        const v = e.values.items;
        std.mem.sort(f64, v, {}, std.sort.asc(f64));
        const med = if (v.len % 2 == 1) v[v.len / 2] else (v[v.len / 2 - 1] + v[v.len / 2]) / 2;
        try stats.append(arena, .{ .key = key, .unit = e.unit, .runs = v.len, .median = med, .min = v[0], .max = v[v.len - 1] });
    }

    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buf);
    const out = &stdout.interface;
    try out.writeAll("device,probe,variant,x,y,unit,runs,median,min,max,spread\n");
    for (stats.items) |s| try out.print("{s},{s},{d},{d:.4},{d:.4},{d:.4},{d:.4}\n", .{ s.key, s.unit, s.runs, s.median, s.min, s.max, s.spread() });
    try out.flush();

    // Summary: rows per spread band for each device/probe.
    const bands = [_]f64{ 0.02, 0.05, 0.10, 0.25 };
    std.debug.print("{s:<22} {s:>5} {s:>6} {s:>6} {s:>6} {s:>6} {s:>6}\n", .{ "device/probe", "rows", "<2%", "<5%", "<10%", "<25%", ">=25%" });
    var groups: std.array_hash_map.String([bands.len + 1]u32) = .empty;
    for (stats.items) |s| {
        var parts = std.mem.splitScalar(u8, s.key, ',');
        const dev = parts.next().?;
        const probe = parts.next().?;
        const g = try std.fmt.allocPrint(arena, "{s}/{s}", .{ dev, probe });
        const gop = try groups.getOrPut(arena, g);
        if (!gop.found_existing) gop.value_ptr.* = @splat(0);
        const band = for (bands, 0..) |b, i| {
            if (s.spread() < b) break i;
        } else bands.len;
        gop.value_ptr[band] += 1;
    }
    for (groups.keys(), groups.values()) |g, c| {
        var total: u32 = 0;
        for (c) |n| total += n;
        std.debug.print("{s:<22} {d:>5} {d:>6} {d:>6} {d:>6} {d:>6} {d:>6}\n", .{ g, total, c[0], c[1], c[2], c[3], c[4] });
    }

    // The most variable rows.
    std.mem.sort(Stat, stats.items, {}, struct {
        fn more(_: void, a: Stat, b: Stat) bool {
            return a.spread() > b.spread();
        }
    }.more);
    std.debug.print("\nmost variable:\n", .{});
    for (stats.items[0..@min(25, stats.items.len)]) |s| {
        std.debug.print("  {s:<52} {d:>6.0}%  median {d:.4} {s} [{d:.4} .. {d:.4}]\n", .{ s.key, s.spread() * 100, s.median, s.unit, s.min, s.max });
    }

    try reportDisturbance(arena, runs);
}

// ---------------------------------------------------------------- disturbance check

/// A marker that moved more than this flags the run; the same value as zstrata's
/// `cpu_probes.disturbed_tolerance` (clean runs move the markers by at most 6% across runs).
const tolerance = 0.10;

const marker_names = [_][]const u8{ "l1-latency", "ram-latency", "fma-per-core" };

/// The `cpu,check` rows of one run file.
const Run = struct {
    path: []const u8,
    /// The run's own verdict (`cpu,check,disturbed`), null in runs made before the check.
    flagged: ?bool = null,
    worst: ?f64 = null,
    /// Marker values at each checkpoint, by `marker_names`.
    marks: [marker_names.len]std.ArrayList(f64) = @splat(.empty),

    fn add(self: *Run, arena: std.mem.Allocator, variant: []const u8, value: f64) !void {
        if (std.mem.eql(u8, variant, "disturbed")) self.flagged = value != 0;
        if (std.mem.eql(u8, variant, "worst-move")) self.worst = value;
        for (marker_names, &self.marks) |n, *m| {
            if (std.mem.eql(u8, variant, n)) try m.append(arena, value);
        }
    }
};

/// Median of `xs` (sorted in place); null when empty.
fn medianOf(xs: []f64) ?f64 {
    if (xs.len == 0) return null;
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return if (xs.len % 2 == 1) xs[xs.len / 2] else (xs[xs.len / 2 - 1] + xs[xs.len / 2]) / 2;
}

/// How far `v` is from `ref`, as a fraction, the same either way (2x and 0.5x are both 1.0).
/// Infinite when either is not positive (a broken timer), instead of dividing by it.
fn moveFrom(v: f64, ref: f64) f64 {
    if (!(v > 0) or !(ref > 0)) return std.math.inf(f64);
    return @max(v / ref, ref / v) - 1;
}

/// One line per run: its own flag and largest within-run move, and how far its markers sit
/// from the median of all runs given (which catches a run disturbed the whole way through,
/// where its own checkpoints agree with each other). With half the runs disturbed the median
/// itself moves, so read it next to the runs' own flags.
fn reportDisturbance(arena: std.mem.Allocator, runs: []Run) !void {
    std.debug.print("\ndisturbance check (tolerance {d:.0}%):\n", .{tolerance * 100});
    // Per run, the median of each marker over its checkpoints; then the median over runs.
    const per_run = try arena.alloc([marker_names.len]?f64, runs.len);
    var refs: [marker_names.len]?f64 = undefined;
    for (0..marker_names.len) |i| {
        var col: std.ArrayList(f64) = .empty;
        for (runs, per_run) |*r, *p| {
            p[i] = medianOf(r.marks[i].items);
            if (p[i]) |v| try col.append(arena, v);
        }
        refs[i] = medianOf(col.items);
    }
    for (runs, per_run) |r, p| {
        if (r.flagged == null) {
            std.debug.print("  {s}: no check rows (made before the check, or with --only/--no-check)\n", .{r.path});
            continue;
        }
        var suspect = r.flagged.?;
        std.debug.print("  {s}: own flag {s}, worst move within the run {d:.1}%; vs all runs:", .{ r.path, if (r.flagged.?) "DISTURBED" else "clean", (r.worst orelse 0) * 100 });
        for (marker_names, p, refs) |n, v, ref| {
            const val = v orelse continue;
            const move = moveFrom(val, ref orelse continue);
            if (move > tolerance) suspect = true;
            std.debug.print(" {s} {d:.1}%", .{ n, move * 100 });
        }
        std.debug.print("{s}\n", .{if (suspect) "  <- SUSPECT" else ""});
    }
}

test "spread: moveFrom is symmetric and guards non-positive values" {
    try std.testing.expectEqual(@as(f64, 0), moveFrom(5, 5));
    try std.testing.expectApproxEqRel(@as(f64, 1), moveFrom(2, 1), 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 1), moveFrom(1, 2), 1e-12);
    try std.testing.expect(std.math.isInf(moveFrom(0, 1)));
    try std.testing.expect(std.math.isInf(moveFrom(1, -1)));
    try std.testing.expect(std.math.isInf(moveFrom(std.math.nan(f64), 1)));
}

test "spread: medianOf" {
    try std.testing.expectEqual(@as(?f64, null), medianOf(&.{}));
    var odd = [_]f64{ 3, 1, 2 };
    try std.testing.expectEqual(@as(?f64, 2), medianOf(&odd));
    var even = [_]f64{ 4, 1, 3, 2 };
    try std.testing.expectEqual(@as(?f64, 2.5), medianOf(&even));
}

test "spread: Run collects check rows by name" {
    var a = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer a.deinit();
    var r: Run = .{ .path = "x" };
    try r.add(a.allocator(), "l1-latency", 1.0);
    try r.add(a.allocator(), "l1-latency", 3.0);
    try r.add(a.allocator(), "fma-per-core", 2.5);
    try r.add(a.allocator(), "disturbed", 1);
    try r.add(a.allocator(), "worst-move", 2.0);
    try r.add(a.allocator(), "other", 9);
    try std.testing.expectEqual(@as(usize, 2), r.marks[0].items.len);
    try std.testing.expectEqual(@as(usize, 0), r.marks[1].items.len);
    try std.testing.expectEqual(@as(usize, 1), r.marks[2].items.len);
    try std.testing.expectEqual(@as(?bool, true), r.flagged);
    try std.testing.expectEqual(@as(?f64, 2.0), r.worst);
}
