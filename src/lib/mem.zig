// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Big buffers the way the hardware wants them: page-aligned (so 64-byte vector loads never
//! straddle two cache lines) and, on Linux, on 2 MiB pages.
//!
//! Why this exists (docs/measurements.md, "Linux gaps: ours or the OS?"):
//!   - glibc's `malloc` returns big blocks 16 bytes into a cache line (its header comes
//!     first), so every 64-byte load from such a block touches two lines: one thread reads an
//!     L1-sized set at 334 GB/s instead of 648. Windows' heap hands out big blocks
//!     page-aligned.
//!   - Linux here gives 2 MiB pages only to memory that asks (`madvise`); with 4 KiB pages a
//!     first write runs at 8.4 GB/s instead of 22-24, and random reads pay more TLB misses.

const std = @import("std");
const builtin = @import("builtin");

pub const page = std.heap.page_size_min;
pub const huge_page = 2 << 20;
const is_linux = builtin.os.tag == .linux;

pub const Pages = enum {
    /// 2 MiB pages where the OS gives them on request (Linux); the OS default elsewhere.
    huge,
    /// Explicitly not: 4 KiB pages even where huge pages are the default. For comparisons.
    small,
};

/// `n` items of fresh page-aligned memory straight from the OS; contents undefined. From 2 MiB up on Linux
/// the block starts on a 2 MiB boundary and is advised per `pages`. Free with `freeBig`.
pub fn allocBig(comptime T: type, n: usize, pages: Pages) error{OutOfMemory}![]align(page) T {
    comptime std.debug.assert(@alignOf(T) <= page);
    const bytes = std.math.mul(usize, n, @sizeOf(T)) catch return error.OutOfMemory;
    if (!is_linux or bytes < huge_page) {
        return std.heap.page_allocator.alignedAlloc(T, .fromByteUnits(page), n);
    }
    const linux = std.os.linux;
    const len = std.mem.alignForward(usize, bytes, page);
    // Over-map by one huge page, then give back what lies before the first 2 MiB boundary and
    // after the end, so `freeBig` can unmap exactly [ptr, ptr + len).
    const span = len + huge_page;
    const r = linux.mmap(null, span, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    if (linux.errno(r) != .SUCCESS) return error.OutOfMemory;
    const base = std.mem.alignForward(usize, r, huge_page);
    if (base > r) _ = linux.munmap(@ptrFromInt(r), base - r);
    if (base + len < r + span) _ = linux.munmap(@ptrFromInt(base + len), r + span - (base + len));
    const ptr: [*]align(page) u8 = @ptrFromInt(base);
    // Advice only: if the kernel refuses, the memory still works, on its default pages.
    _ = linux.madvise(ptr, len, switch (pages) {
        .huge => linux.MADV.HUGEPAGE,
        .small => linux.MADV.NOHUGEPAGE,
    });
    return @as([*]align(page) T, @ptrCast(ptr))[0..n];
}

pub fn freeBig(comptime T: type, mem: []align(page) T) void {
    const bytes = mem.len * @sizeOf(T);
    if (!is_linux or bytes < huge_page) return std.heap.page_allocator.free(mem);
    _ = std.os.linux.munmap(@ptrCast(mem.ptr), std.mem.alignForward(usize, bytes, page));
}

test "allocBig: aligned, writable, both sizes and page kinds" {
    for ([_]usize{ 1000, (3 << 20) + 5 }) |n| for ([_]Pages{ .huge, .small }) |p| {
        const m = try allocBig(u32, n, p);
        defer freeBig(u32, m);
        try std.testing.expectEqual(n, m.len);
        try std.testing.expectEqual(@as(usize, 0), @intFromPtr(m.ptr) % page);
        if (is_linux and n * 4 >= huge_page) try std.testing.expectEqual(@as(usize, 0), @intFromPtr(m.ptr) % huge_page);
        m[0] = 5;
        m[n - 1] = 7;
        try std.testing.expectEqual(@as(u32, 5), m[0]);
        try std.testing.expectEqual(@as(u32, 7), m[n - 1]);
    };
}
