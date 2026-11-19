// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! zstrata as a library: the pieces other projects build on, each one the form that measured
//! fastest in zstrata-bench.
//!
//! In another project's build.zig:
//!
//!     const zstrata = b.dependency("zstrata", .{ .target = target, .optimize = optimize, .@"lib-only" = true });
//!     exe.root_module.addImport("zstrata", zstrata.module("zstrata"));
//!
//! `lib-only` leaves out the measuring program, so the Vulkan packages and the shader compiler
//! are not needed.

/// The thread pool: `parallelFor` over a range, helpers that sleep when idle.
pub const Pool = @import("cpu/pool.zig").Pool;
/// Vector width for the build's CPU and a multiply-add that is only fused where that is fast.
pub const simd = @import("lib/simd.zig");
/// Radix sorts for integers, floats, and row indices by a float column.
pub const sort = @import("lib/sort.zig");

test {
    _ = @import("cpu/pool.zig");
    _ = simd;
    _ = sort;
}
