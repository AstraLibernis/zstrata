// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! What a hot loop needs to know about the CPU it is being built for.
//!
//! Zig 0.17 does not turn plain loops into vector code, so loops that matter are written with
//! `@Vector`. These helpers keep such code fast on the machine it is built for and correct and
//! reasonable on any other (zstrata-bench, docs/measurements.md, "Vectorisation").

const std = @import("std");
const builtin = @import("builtin");

/// How many `T` to process at a time: what `std.simd` suggests for the CPU this build targets
/// (16 floats with AVX-512, 8 with AVX2, 4 on a plain 64-bit CPU), or 4 when it has no
/// suggestion. A fixed 16 also works everywhere; this just avoids the split on narrower CPUs.
pub fn width(comptime T: type) comptime_int {
    return std.simd.suggestVectorLength(T) orelse 4;
}

/// Whether the CPU this build targets has a fused multiply-add instruction. Without one,
/// `@mulAdd` still gives the right answer, but through a library routine per value (measured
/// 20x to 35x slower than a separate multiply and add).
pub const fused_is_an_instruction = builtin.cpu.has(.x86, .fma) or builtin.cpu.arch.isAARCH64();

/// `a * b + c`: fused where the CPU has the instruction, a separate multiply and add where it
/// does not. `T` is a float or a vector of floats. The two forms can differ in the last bit.
pub inline fn mulAdd(comptime T: type, a: T, b: T, c: T) T {
    return if (fused_is_an_instruction) @mulAdd(T, a, b, c) else a * b + c;
}

test "simd: width is a usable vector length and mulAdd computes a * b + c" {
    const n = width(f32);
    try std.testing.expect(n >= 4 and n <= 64);
    const V = @Vector(n, f32);
    const a: V = @splat(3);
    const b: V = @splat(0.5);
    const c: V = @splat(-1);
    const got: [n]f32 = mulAdd(V, a, b, c);
    for (got) |x| try std.testing.expectEqual(@as(f32, 0.5), x);
    try std.testing.expectEqual(@as(f64, 7), mulAdd(f64, 2, 3, 1));
}
