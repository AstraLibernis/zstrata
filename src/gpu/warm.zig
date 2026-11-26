// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! Keep the GPU's clock up between short jobs.
//!
//! On Linux (amdgpu, power level `auto`) the RX 9070 XT's clock follows the load: about
//! 1.6 GHz when a job starts, then, only after ~0.5 s of unbroken work, a climb to 3.2-3.4 GHz;
//! ~0.1 s of light load and it is back down. A job shorter than that runs at half speed
//! (docs/measurements.md, "Linux: the GPU clock"). AMD's Windows driver does not show this.
//!
//! `Warmer` keeps one lane of the GPU busy from a helper thread, on the second compute queue
//! (a different kernel ring, so it runs beside the measured work instead of in front of it).
//! One lane out of thousands; what it buys is the clock. Its jobs are long (~0.2 s) because
//! every job boundary flushes the GPU's caches (shaders/spin.comp), so stopping the warmer
//! waits for the running job: up to ~0.2 s.

const std = @import("std");
const gpu = @import("../gpu.zig");
const vk = gpu.vk;
const host = @import("../cpu/host.zig");

const spin_spv align(@alignOf(u32)) = @embedFile("spin.spv").*;

pub const Warmer = struct {
    ctx: *const gpu.Context,
    queue: vk.Queue,
    pipe: gpu.ComputePipeline,
    /// { rounds, done, x } as shaders/spin.comp declares it; CPU-visible so `rounds` can be set.
    ctl: gpu.Buffer,
    pool: vk.CommandPool,
    cmd: vk.CommandBuffer,
    fence: vk.Fence,
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    /// ns per round of the latest job (0 before the first): it shrinks as the clock climbs.
    round_ns: std.atomic.Value(u64) = .init(0),
    rounds: u32 = first_rounds,
    thread: ?std.Thread = null,

    /// One job aims at this long. Long enough that job boundaries are rare, far under the
    /// ~1.5 s at which a single job risks a GPU reset (a 4 s job stalled the GPU's other queue).
    const job_ns = 200 * std.time.ns_per_ms;
    const first_rounds = 256;
    const max_rounds = 1 << 22;

    /// Null when the driver has no second compute queue. `deinit` waits for the running job. The result must not move once
    /// `start` has run (the thread holds its address).
    pub fn init(ctx: *const gpu.Context) !?Warmer {
        const queue = ctx.aux_queue orelse return null;
        const vkd = ctx.vkd;
        var ctl = try gpu.Buffer.init(ctx, 4096, .{ .storage_buffer = true }, .{ .host_visible = true, .host_coherent = true }, .{});
        errdefer ctl.deinit(ctx);
        @memset(ctl.mapped.?[0..16], 0);
        var pipe = try gpu.ComputePipeline.init(ctx, &spin_spv, 1, 0);
        errdefer pipe.deinit(ctx);
        pipe.bindBuffers(ctx, &.{ctl.handle});
        const pool = try vkd.createCommandPool(ctx.device, &.{ .queue_family_index = ctx.aux_family }, null);
        errdefer vkd.destroyCommandPool(ctx.device, pool, null);
        var cmd: vk.CommandBuffer = undefined;
        try vkd.allocateCommandBuffers(ctx.device, &.{ .command_pool = pool, .level = .primary, .command_buffer_count = 1 }, @ptrCast(&cmd));
        const fence = try vkd.createFence(ctx.device, &.{}, null);
        errdefer vkd.destroyFence(ctx.device, fence, null);
        // Recorded once, resubmitted for the life of the warmer.
        try vkd.beginCommandBuffer(cmd, &.{});
        pipe.dispatch(ctx, cmd, null, .{ 1, 1, 1 });
        try vkd.endCommandBuffer(cmd);
        return .{ .ctx = ctx, .queue = queue, .pipe = pipe, .ctl = ctl, .pool = pool, .cmd = cmd, .fence = fence };
    }

    /// Start the helper thread and wait until the clock has stopped climbing: until the
    /// time per round has not improved by 1% for `settled_ms`, at most `settle_limit_ms`.
    pub fn start(self: *Warmer) !void {
        self.thread = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, loop, .{self});
        var best: u64 = std.math.maxInt(u64);
        var quiet_ms: u32 = 0;
        var waited_ms: u32 = 0;
        while (quiet_ms < settled_ms and waited_ms < settle_limit_ms and !self.failed.load(.acquire)) : (waited_ms += 20) {
            host.sleepMs(20);
            const t = self.round_ns.load(.acquire);
            if (t != 0 and t < best - best / 100) {
                best = t;
                quiet_ms = 0;
            } else if (t != 0) quiet_ms += 20;
        }
    }

    const settled_ms = 800;
    const settle_limit_ms = 5000;

    /// False once a submit or wait has failed; the thread has stopped and the clock is no
    /// longer held.
    pub fn running(self: *const Warmer) bool {
        return self.thread != null and !self.failed.load(.acquire);
    }

    fn loop(self: *Warmer) void {
        while (!self.stop.load(.acquire)) self.once() catch {
            self.failed.store(true, .release);
            return;
        };
    }

    fn once(self: *Warmer) !void {
        const vkd = self.ctx.vkd;
        self.word(0).* = self.rounds;
        const t0 = host.nowNs();
        const info = [_]vk.CommandBufferSubmitInfo{.{ .command_buffer = self.cmd, .device_mask = 0 }};
        try vkd.queueSubmit2(self.queue, &.{.{ .command_buffer_info_count = info.len, .p_command_buffer_infos = &info }}, self.fence);
        _ = try vkd.waitForFences(self.ctx.device, &.{self.fence}, .true, std.math.maxInt(u64));
        try vkd.resetFences(self.ctx.device, &.{self.fence});
        const per_round = @max(1, (host.nowNs() - t0) / self.rounds);
        self.round_ns.store(per_round, .release);
        self.rounds = @intCast(std.math.clamp(job_ns / per_round, first_rounds, max_rounds));
    }

    /// Word `i` of the control buffer.
    fn word(self: *Warmer, i: usize) *u32 {
        return &@as([*]u32, @ptrCast(@alignCast(self.ctl.mapped.?)))[i];
    }

    pub fn deinit(self: *Warmer) void {
        self.stop.store(true, .release);
        if (self.thread) |t| t.join();
        const ctx = self.ctx;
        ctx.vkd.destroyFence(ctx.device, self.fence, null);
        ctx.vkd.destroyCommandPool(ctx.device, self.pool, null);
        self.pipe.deinit(ctx);
        self.ctl.deinit(ctx);
        self.* = undefined;
    }
};

test "gpu warmer: keeps jobs running on the second queue; stopping waits for one job at most" {
    @import("../report.zig").progress = false;
    var ctx = try gpu.Context.init(std.testing.allocator, .{ .validation = true });
    defer ctx.deinit();
    var warmer = (try Warmer.init(&ctx)) orelse return error.SkipZigTest;
    {
        // The thread must be stopped before the device goes, on a failed check too.
        errdefer warmer.deinit();
        try warmer.start();
        try std.testing.expect(warmer.running());
        try std.testing.expect(warmer.round_ns.load(.acquire) > 0);
    }
    const t0 = host.nowNs();
    warmer.deinit();
    try std.testing.expect(host.nowNs() - t0 < 3 * Warmer.job_ns);
    try std.testing.expectEqual(@as(u32, 0), gpu.validation_messages.load(.monotonic));
}
