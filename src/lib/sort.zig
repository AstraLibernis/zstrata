// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Radix sorts: the fastest one-thread sort measured for fixed-width keys (zstrata-bench D3),
//! for unsigned integers, for floats, and for row indices ordered by a float column.
//!
//! All are stable, sort ascending, and take their working memory from the caller or an
//! allocator; none keeps anything after returning.

const std = @import("std");

/// Below this many items a comparison sort is used: the radix passes have a fixed cost.
const small = 256;

/// Sorts `items` ascending by `key_bits` bits of each item starting at bit `shift`, taking
/// `digit_bits` bits per pass. Stable. `scratch.len` must equal `items.len`. The result is in
/// `items`.
///
/// `T` is an unsigned integer. With `shift` and `key_bits` covering only part of it, the rest
/// of each item rides along as a payload.
pub fn radix(
    comptime T: type,
    comptime digit_bits: comptime_int,
    comptime shift: comptime_int,
    comptime key_bits: comptime_int,
    items: []T,
    scratch: []T,
) void {
    std.debug.assert(scratch.len == items.len);
    const passes = (key_bits + digit_bits - 1) / digit_bits;
    const buckets = 1 << digit_bits;

    // One read of the data counts every pass's digits.
    var counts: [passes][buckets]usize = @splat(@splat(0));
    for (items) |item| {
        const key = item >> shift;
        inline for (0..passes) |p| counts[p][@intCast((key >> (p * digit_bits)) & (buckets - 1))] += 1;
    }

    var src = items;
    var dst = scratch;
    inline for (0..passes) |p| {
        // Counts become each bucket's first position.
        var at: usize = 0;
        for (&counts[p]) |*c| {
            const n = c.*;
            c.* = at;
            at += n;
        }
        for (src) |item| {
            const digit: usize = @intCast(((item >> shift) >> (p * digit_bits)) & (buckets - 1));
            dst[counts[p][digit]] = item;
            counts[p][digit] += 1;
        }
        std.mem.swap([]T, &src, &dst);
    }
    // After an odd number of passes the sorted data sits in `scratch`.
    if (passes % 2 == 1) @memcpy(items, scratch);
}

fn Bits(comptime F: type) type {
    return switch (F) {
        f32 => u32,
        f64 => u64,
        else => @compileError("floats are f32 or f64"),
    };
}

/// An unsigned integer that orders the same way as the float: flip every bit of a negative
/// value, set the top bit of the rest.
inline fn toKey(comptime F: type, x: F) Bits(F) {
    const U = Bits(F);
    const top = @as(U, 1) << (@bitSizeOf(U) - 1);
    const b: U = @bitCast(x);
    return if (b & top != 0) ~b else b | top;
}

inline fn fromKey(comptime F: type, k: Bits(F)) F {
    const U = Bits(F);
    const top = @as(U, 1) << (@bitSizeOf(U) - 1);
    return @bitCast(if (k & top != 0) k & ~top else ~k);
}

/// Sorts `values` ascending. `scratch.len` must equal `values.len`. -0 sorts before +0. The
/// values are expected to hold no NaN; any that are present end up at the two ends, by sign.
///
/// A pass whose byte is the same in every value is skipped, so columns of small whole numbers
/// or of a narrow range cost fewer passes.
pub fn floats(comptime F: type, values: []F, scratch: []F) void {
    std.debug.assert(scratch.len == values.len);
    const n = values.len;
    if (n < small) {
        const Less = struct {
            fn less(_: void, p: F, q: F) bool {
                return toKey(F, p) < toKey(F, q);
            }
        };
        return std.mem.sort(F, values, {}, Less.less);
    }
    const U = Bits(F);
    const bytes = @sizeOf(U);
    var a: []U = @ptrCast(values);
    var b: []U = @ptrCast(scratch);
    var counts: [bytes][256]usize = @splat(@splat(0));
    for (a) |*v| {
        const k = toKey(F, @bitCast(v.*));
        v.* = k;
        inline for (0..bytes) |p| counts[p][@as(u8, @truncate(k >> (8 * p)))] += 1;
    }
    inline for (0..bytes) |p| {
        const c = &counts[p];
        if (c[@as(u8, @truncate(a[0] >> (8 * p)))] != n) {
            var at: usize = 0;
            for (c) |*x| {
                const t = x.*;
                x.* = at;
                at += t;
            }
            for (a) |k| {
                const d: u8 = @truncate(k >> (8 * p));
                b[c[d]] = k;
                c[d] += 1;
            }
            std.mem.swap([]U, &a, &b);
        }
    }
    // `a` is `values` or `scratch`, whichever the last pass wrote; the result goes to `values`.
    for (a, 0..) |k, i| values[i] = fromKey(F, k);
}

/// Reorders `idx` so that `values[idx[0]], values[idx[1]], ...` ascends; rows with equal values
/// keep the order they came in. Every entry of `idx` is a valid position in `values`, and none
/// of those values is NaN. Working memory comes from `gpa` and is freed before returning.
pub fn floatIndex(comptime F: type, gpa: std.mem.Allocator, values: []const F, idx: []u32) !void {
    const n = idx.len;
    if (n < small) {
        const Less = struct {
            v: []const F,
            fn less(c: @This(), p: u32, q: u32) bool {
                return toKey(F, c.v[p]) < toKey(F, c.v[q]);
            }
        };
        return std.mem.sort(u32, idx, Less{ .v = values }, Less.less);
    }
    const U = Bits(F);
    const bytes = @sizeOf(U);
    const Rec = struct { k: U, i: u32 };
    const mem = try gpa.alloc(Rec, 2 * n);
    defer gpa.free(mem);
    var a = mem[0..n];
    var b = mem[n..];
    var counts: [bytes][256]usize = @splat(@splat(0));
    for (a, idx) |*r, i| {
        const k = toKey(F, values[i]);
        r.* = .{ .k = k, .i = i };
        inline for (0..bytes) |p| counts[p][@as(u8, @truncate(k >> (8 * p)))] += 1;
    }
    inline for (0..bytes) |p| {
        const c = &counts[p];
        if (c[@as(u8, @truncate(a[0].k >> (8 * p)))] != n) {
            var at: usize = 0;
            for (c) |*x| {
                const t = x.*;
                x.* = at;
                at += t;
            }
            for (a) |r| {
                const d: u8 = @truncate(r.k >> (8 * p));
                b[c[d]] = r;
                c[d] += 1;
            }
            std.mem.swap([]Rec, &a, &b);
        }
    }
    for (a, idx) |r, *i| i.* = r.i;
}

// ---------------------------------------------------------------- tests

const testing = std.testing;
const edge_sizes = [_]usize{ 0, 1, 2, 3, 255, 256, 257, 5000 };

fn expectRadix(comptime digit_bits: comptime_int, n: usize, seed: u64) !void {
    const gpa = testing.allocator;
    const a = try gpa.alloc(u32, n);
    defer gpa.free(a);
    const b = try gpa.alloc(u32, n);
    defer gpa.free(b);
    const scratch = try gpa.alloc(u32, n);
    defer gpa.free(scratch);
    var prng = std.Random.DefaultPrng.init(seed);
    for (a) |*x| x.* = prng.random().int(u32);
    // Extremes and repeats, where a wrong digit mask or count would show.
    if (n > 4) {
        a[0] = 0;
        a[1] = std.math.maxInt(u32);
        a[2] = a[3];
    }
    @memcpy(b, a);
    std.mem.sort(u32, b, {}, std.sort.asc(u32));
    radix(u32, digit_bits, 0, 32, a, scratch);
    try testing.expectEqualSlices(u32, b, a);
}

test "sort: radix on u32 keys, even and odd pass counts, edge sizes" {
    for (edge_sizes) |n| {
        try expectRadix(8, n, n + 1); // 4 passes: ends in `items`
        try expectRadix(11, n, n + 2); // 3 passes: copied back from scratch
    }
}

test "sort: radix on u64 records is stable and carries the payload" {
    const gpa = testing.allocator;
    const n = 4000;
    const a = try gpa.alloc(u64, n);
    defer gpa.free(a);
    const scratch = try gpa.alloc(u64, n);
    defer gpa.free(scratch);
    var prng = std.Random.DefaultPrng.init(9);
    // Key in the high half (few distinct, so ties are common), arrival order in the low half.
    for (a, 0..) |*x, i| x.* = (@as(u64, prng.random().uintLessThan(u32, 50)) << 32) | i;
    radix(u64, 8, 32, 32, a, scratch);
    for (a[0 .. n - 1], a[1..]) |p, q| {
        try testing.expect(p >> 32 <= q >> 32);
        if (p >> 32 == q >> 32) try testing.expect(@as(u32, @truncate(p)) < @as(u32, @truncate(q)));
    }
}

/// Values chosen to exercise every pass: wide range, small whole numbers, tiny magnitudes,
/// negatives, both zeros, infinities and repeats.
fn fillFloats(comptime F: type, x: []F, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (x, 0..) |*v, i| v.* = switch (i % 6) {
        0 => r.floatNorm(F) * 1e6,
        1 => @floatFromInt(r.intRangeAtMost(i32, -50, 50)),
        2 => r.float(F) * std.math.floatMin(F) * 8,
        3 => -r.float(F),
        4 => if (r.boolean()) std.math.inf(F) else -std.math.inf(F),
        else => if (r.boolean()) @as(F, 0.0) else -@as(F, 0.0),
    };
}

test "sort: floats gives the comparison sort's order, f32 and f64" {
    const gpa = testing.allocator;
    inline for (.{ f32, f64 }) |F| {
        for (edge_sizes ++ [_]usize{50_000}) |n| {
            const x = try gpa.alloc(F, n);
            defer gpa.free(x);
            const want = try gpa.alloc(F, n);
            defer gpa.free(want);
            const scratch = try gpa.alloc(F, n);
            defer gpa.free(scratch);
            fillFloats(F, x, n + 3);
            @memcpy(want, x);
            std.mem.sort(F, want, {}, std.sort.asc(F));
            floats(F, x, scratch);
            // Compared as numbers: the two zeros are equal there, and radix puts -0 first.
            for (want, x) |w, got| try testing.expectEqual(w, got);
            for (x[0..n -| 1], x[@min(1, n)..]) |p, q| {
                if (p == 0 and q == 0) try testing.expect(std.math.signbit(p) or !std.math.signbit(q));
            }
        }
    }
}

test "sort: floats skips passes on whole numbers and still sorts them" {
    const gpa = testing.allocator;
    const n = 3000;
    const x = try gpa.alloc(f64, n);
    defer gpa.free(x);
    const scratch = try gpa.alloc(f64, n);
    defer gpa.free(scratch);
    // 0..999 as doubles: the low five bytes are zero in every value.
    for (x, 0..) |*v, i| v.* = @floatFromInt((i * 7919) % 1000);
    floats(f64, x, scratch);
    for (x, 0..) |v, i| try testing.expectEqual(@as(f64, @floatFromInt(i / 3)), v);
}

test "sort: floatIndex orders rows by value and keeps ties in arrival order" {
    const gpa = testing.allocator;
    inline for (.{ f32, f64 }) |F| {
        for (edge_sizes) |n| {
            const x = try gpa.alloc(F, n + 10);
            defer gpa.free(x);
            fillFloats(F, x, n + 5);
            // A subset of the rows, in increasing row order.
            var idx: std.ArrayList(u32) = .empty;
            defer idx.deinit(gpa);
            for (0..n + 10) |i| if (i % 11 != 0 and idx.items.len < n) try idx.append(gpa, @intCast(i));
            const want = try gpa.dupe(u32, idx.items);
            defer gpa.free(want);
            const Less = struct {
                v: []const F,
                fn less(c: @This(), p: u32, q: u32) bool {
                    // The radix order: by value, -0 before +0.
                    return toKey(F, c.v[p]) < toKey(F, c.v[q]);
                }
            };
            std.mem.sort(u32, want, Less{ .v = x }, Less.less); // stable
            try floatIndex(F, gpa, x, idx.items);
            try testing.expectEqualSlices(u32, want, idx.items);
            for (idx.items[0 .. idx.items.len -| 1], idx.items[@min(1, idx.items.len)..]) |p, q| {
                try testing.expect(x[p] <= x[q]);
                if (x[p] == x[q] and toKey(F, x[p]) == toKey(F, x[q])) try testing.expect(p < q);
            }
        }
    }
}

test "sort: floatIndex gives back every byte it borrowed, also when memory runs out" {
    const x = try testing.allocator.alloc(f64, 1000);
    defer testing.allocator.free(x);
    fillFloats(f64, x, 1);
    const idx = try testing.allocator.alloc(u32, 1000);
    defer testing.allocator.free(idx);
    for (idx, 0..) |*v, i| v.* = @intCast(i);
    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    try testing.expectError(error.OutOfMemory, floatIndex(f64, failing.allocator(), x, idx));
    for (idx, 0..) |v, i| try testing.expectEqual(@as(u32, @intCast(i)), v);
}
