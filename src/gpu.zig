// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis

//! The setup every experiment repeats: load the Vulkan loader, create an instance (with the
//! validation layer in Debug builds), pick a GPU, create a device with one compute queue.
//!
//! Switches a user can set: `Options.validation` (default: on in Debug); validation warnings
//! and errors are printed and counted in `validation_messages`; the loader's own notes about
//! the host's layer setup are dropped unless `show_loader_notes` is set (turn it on in a
//! Debug build when a device will not open).

const std = @import("std");
const builtin = @import("builtin");
pub const vk = @import("vulkan");
const host = @import("cpu/host.zig");
const report = @import("report.zig");

const validation_layer: [*:0]const u8 = "VK_LAYER_KHRONOS_validation";

/// The Vulkan loader library. Linux: libvulkan.so.1 through std.DynLib. Windows: vulkan-1.dll,
/// which the GPU driver installs, through LoadLibraryW/GetProcAddress (std.DynLib has no
/// Windows support in Zig 0.17).
const Loader = if (builtin.os.tag == .windows) WinLoader else struct {
    inner: std.DynLib,

    fn open() !@This() {
        return .{ .inner = try std.DynLib.open("libvulkan.so.1") };
    }
    fn lookup(self: *@This(), comptime T: type, name: [:0]const u8) ?T {
        return self.inner.lookup(T, name);
    }
    fn close(self: *@This()) void {
        self.inner.close();
    }
};

const WinLoader = struct {
    module: *anyopaque,

    const kernel32 = struct {
        extern "kernel32" fn LoadLibraryW(name: [*:0]const u16) callconv(.winapi) ?*anyopaque;
        extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.winapi) ?*const anyopaque;
        extern "kernel32" fn FreeLibrary(module: *anyopaque) callconv(.winapi) c_int;
    };

    fn open() !WinLoader {
        const m = kernel32.LoadLibraryW(std.unicode.utf8ToUtf16LeStringLiteral("vulkan-1.dll")) orelse
            return error.VulkanLoaderNotFound;
        return .{ .module = m };
    }
    fn lookup(self: *WinLoader, comptime T: type, name: [:0]const u8) ?T {
        const p = kernel32.GetProcAddress(self.module, name.ptr) orelse return null;
        return @ptrCast(p);
    }
    fn close(self: *WinLoader) void {
        _ = kernel32.FreeLibrary(self.module);
    }
};

/// Validation warnings and errors seen since start. Experiments assert this stays 0.
pub var validation_messages: std.atomic.Value(u32) = .init(0);

/// Print the Vulkan loader's notes about the host's layer setup (Debug builds only,
/// where the messenger is on). Off by default: on the sandbox host they are Steam layer
/// files this account may not read, a registry key it may not read, and a duplicate AMD
/// layer from an older driver, none of which this program can change, and every Debug run
/// printed them as "vulkan loader error". Turn on to diagnose a device that will not open.
pub var show_loader_notes = false;

pub const Options = struct {
    /// Validation layer and debug messenger. Defaults to on in Debug, off in release builds,
    /// because the layer slows every call down.
    validation: bool = builtin.mode == .debug,
    /// Index into the physical-device list; null picks the first discrete GPU.
    device_index: ?usize = null,
    /// Extra device extensions, e.g. "VK_EXT_shader_atomic_float".
    device_extensions: []const [*:0]const u8 = &.{},
    /// Device extensions enabled only when the driver offers them (a probe that needs one
    /// checks `Context.enabled` and skips itself otherwise). At most `max_optional`.
    optional_extensions: []const [*:0]const u8 = &.{},
    /// Extra feature structs to enable, chained after the Vulkan 1.3 features
    /// (a vk.PhysicalDevice*Features struct; set its own p_next to chain more).
    device_features: ?*anyopaque = null,
};

pub const Context = struct {
    lib: Loader,
    vkb: vk.BaseWrapper,
    vki: vk.InstanceWrapper,
    vkd: vk.DeviceWrapper,
    instance: vk.Instance,
    messenger: vk.DebugUtilsMessengerEXT,
    physical: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    device: vk.Device,
    queue: vk.Queue,
    queue_family: u32,
    /// Bits a timestamp on the compute queue carries (64 on RDNA); fewer means it wraps.
    timestamp_bits: u32,
    optional_on: [max_optional][*:0]const u8 = undefined,
    optional_len: usize = 0,

    pub const max_optional = 8;

    /// Whether `name` was enabled on this device (required or optional).
    pub fn enabled(self: *const Context, name: [*:0]const u8) bool {
        for (self.optional_on[0..self.optional_len]) |n| {
            if (std.mem.eql(u8, std.mem.span(n), std.mem.span(name))) return true;
        }
        return false;
    }

    pub fn init(gpa: std.mem.Allocator, opts: Options) !Context {
        var lib = try Loader.open();
        errdefer lib.close();
        const get_proc = lib.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse
            return error.NoGetInstanceProcAddr;

        const vkb = vk.BaseWrapper.load(get_proc);

        const validation = opts.validation and try hasLayer(gpa, vkb, validation_layer);
        if (opts.validation and !validation) std.log.warn("validation layer not found; {s}", .{
            if (builtin.os.tag == .windows)
                "set VK_ADD_LAYER_PATH to the folder holding VkLayer_khronos_validation.json (the build's run and test steps use %USERPROFILE%\\.local\\vk-layers)"
            else
                "source tools/env.sh",
        });

        const layers: []const [*:0]const u8 = if (validation) &.{validation_layer} else &.{};
        const exts: []const [*:0]const u8 = if (validation)
            &.{ vk.instance_extensions.ext_debug_utils.name, vk.instance_extensions.ext_layer_settings.name }
        else
            &.{};
        const messenger_info: vk.DebugUtilsMessengerCreateInfoEXT = .{
            .message_severity = .{ .warning_ext = true, .error_ext = true },
            .message_type = .{ .general_ext = true, .validation_ext = true, .performance_ext = true },
            .pfn_user_callback = debugCallback,
        };
        // Synchronization validation is off by default even with the layer on, and even when on
        // it ignores what shaders read and write unless the shader-access setting is on too.
        // Without both, a missing barrier between a copy and a dispatch goes unreported.
        const on: vk.Bool32 = .true;
        const settings = [_]vk.LayerSettingEXT{
            .{ .p_layer_name = validation_layer, .p_setting_name = "validate_sync", .type = .bool32_ext, .value_count = 1, .p_values = &on },
            .{ .p_layer_name = validation_layer, .p_setting_name = "syncval_shader_accesses_heuristic", .type = .bool32_ext, .value_count = 1, .p_values = &on },
        };
        const layer_settings: vk.LayerSettingsCreateInfoEXT = .{
            // Chaining the messenger info also reports problems inside createInstance itself.
            .p_next = &messenger_info,
            .setting_count = settings.len,
            .p_settings = &settings,
        };
        const instance = try vkb.createInstance(&.{
            .p_next = if (validation) &layer_settings else null,
            .p_application_info = &.{
                .p_application_name = "zstrata",
                .application_version = 0,
                .engine_version = 0,
                .api_version = vk.API_VERSION_1_3.toU32(),
            },
            .enabled_layer_count = @intCast(layers.len),
            .pp_enabled_layer_names = layers.ptr,
            .enabled_extension_count = @intCast(exts.len),
            .pp_enabled_extension_names = exts.ptr,
        }, null);
        const vki = vk.InstanceWrapper.load(instance, get_proc);
        errdefer vki.destroyInstance(instance, null);

        const messenger: vk.DebugUtilsMessengerEXT = if (validation)
            try vki.createDebugUtilsMessengerEXT(instance, &messenger_info, null)
        else
            .null;
        errdefer if (messenger != .null) vki.destroyDebugUtilsMessengerEXT(instance, messenger, null);

        const physicals = try vki.enumeratePhysicalDevicesAlloc(instance, gpa);
        defer gpa.free(physicals);
        const physical = try pickDevice(vki, physicals, opts.device_index);
        const props = vki.getPhysicalDeviceProperties(physical);

        const family = try computeQueueFamily(gpa, vki, physical);
        const queue_family = family.index;

        // Required extensions, then the optional ones this driver offers.
        if (opts.optional_extensions.len > max_optional) return error.TooManyOptionalExtensions;
        var optional_on: [max_optional][*:0]const u8 = undefined;
        var optional_len: usize = 0;
        const offered = try vki.enumerateDeviceExtensionPropertiesAlloc(physical, null, gpa);
        defer gpa.free(offered);
        for (opts.optional_extensions) |want| {
            for (offered) |e| {
                if (std.mem.eql(u8, std.mem.sliceTo(&e.extension_name, 0), std.mem.span(want))) {
                    optional_on[optional_len] = want;
                    optional_len += 1;
                    break;
                }
            }
        }
        const dev_exts = try gpa.alloc([*:0]const u8, opts.device_extensions.len + optional_len);
        defer gpa.free(dev_exts);
        @memcpy(dev_exts[0..opts.device_extensions.len], opts.device_extensions);
        @memcpy(dev_exts[opts.device_extensions.len..], optional_on[0..optional_len]);
        const priority = [_]f32{1.0};
        const queue_info = [_]vk.DeviceQueueCreateInfo{.{
            .queue_family_index = queue_family,
            .queue_count = 1,
            .p_queue_priorities = &priority,
        }};
        // Synchronization2 (core in 1.3, but still opt-in): the barrier, submit and timestamp
        // calls with the clearer stage/access flags.
        var features13: vk.PhysicalDeviceVulkan13Features = .{
            .synchronization_2 = .true,
            .dynamic_rendering = .true, // draw without render-pass objects
            .subgroup_size_control = .true, // let a pipeline pin its subgroup width
            .compute_full_subgroups = .true,
            .p_next = opts.device_features,
        };
        const device = try vki.createDevice(physical, &.{
            .p_next = &features13,
            .queue_create_info_count = queue_info.len,
            .p_queue_create_infos = &queue_info,
            .enabled_extension_count = @intCast(dev_exts.len),
            .pp_enabled_extension_names = dev_exts.ptr,
        }, null);
        const vkd = vk.DeviceWrapper.load(device, vki.dispatch.vkGetDeviceProcAddr.?);
        errdefer vkd.destroyDevice(device, null);

        return .{
            .lib = lib,
            .vkb = vkb,
            .vki = vki,
            .vkd = vkd,
            .instance = instance,
            .messenger = messenger,
            .physical = physical,
            .props = props,
            .device = device,
            .queue = vkd.getDeviceQueue(device, queue_family, 0),
            .queue_family = queue_family,
            .timestamp_bits = family.timestamp_bits,
            .optional_on = optional_on,
            .optional_len = optional_len,
        };
    }

    pub fn deinit(self: *Context) void {
        self.vkd.destroyDevice(self.device, null);
        if (self.messenger != .null)
            self.vki.destroyDebugUtilsMessengerEXT(self.instance, self.messenger, null);
        self.vki.destroyInstance(self.instance, null);
        self.lib.close();
        self.* = undefined;
    }

    pub fn deviceName(self: *const Context) []const u8 {
        return std.mem.sliceTo(&self.props.device_name, 0);
    }

    /// What the driver says this GPU can do, as plain text: which driver answers, the compute
    /// limits the probes depend on, the memory heaps, and the extensions zstrata uses. Two
    /// drivers for the same card differ here (RADV vs AMD's Windows driver), so a run's numbers
    /// should be read next to this.
    pub fn writeDeviceInfo(self: *const Context, gpa: std.mem.Allocator, w: *std.Io.Writer) !void {
        // zeroInit keeps each field's default (s_type) and zeroes the rest; the driver fills them.
        var v13 = std.mem.zeroInit(vk.PhysicalDeviceVulkan13Properties, .{});
        var driver = std.mem.zeroInit(vk.PhysicalDeviceDriverProperties, .{ .p_next = @as(?*anyopaque, @ptrCast(&v13)) });
        var p2: vk.PhysicalDeviceProperties2 = .{ .p_next = &driver, .properties = undefined };
        self.vki.getPhysicalDeviceProperties2(self.physical, &p2);
        const p = p2.properties;
        const l = p.limits;
        const cv = driver.conformance_version;
        try w.print("device       {s}\n", .{self.deviceName()});
        try w.print("driver       {s} ({s}), id {t}, conformance {d}.{d}.{d}.{d}\n", .{ std.mem.sliceTo(&driver.driver_name, 0), std.mem.sliceTo(&driver.driver_info, 0), driver.driver_id, cv.major, cv.minor, cv.subminor, cv.patch });
        // VK_MAKE_API_VERSION layout: major bits 22-28, minor 12-21, patch 0-11.
        const av = p.api_version;
        try w.print("vulkan       {d}.{d}.{d}\n", .{ (av >> 22) & 0x7f, (av >> 12) & 0x3ff, av & 0xfff });
        try w.print("subgroups    {d}..{d} lanes, up to {d} per workgroup\n", .{ v13.min_subgroup_size, v13.max_subgroup_size, v13.max_compute_workgroup_subgroups });
        try w.print("workgroup    {d} invocations, size {d} x {d} x {d}, shared memory {d} KiB\n", .{ l.max_compute_work_group_invocations, l.max_compute_work_group_size[0], l.max_compute_work_group_size[1], l.max_compute_work_group_size[2], l.max_compute_shared_memory_size / 1024 });
        try w.print("buffers      storage range {d} MiB, max buffer {d} MiB, allocations up to {d}\n", .{ l.max_storage_buffer_range >> 20, v13.max_buffer_size >> 20, l.max_memory_allocation_count });
        try w.print("push consts  {d} bytes; timestamp period {d} ns\n", .{ l.max_push_constants_size, l.timestamp_period });
        const mem = self.vki.getPhysicalDeviceMemoryProperties(self.physical);
        for (mem.memory_heaps[0..mem.memory_heap_count], 0..) |h, i| {
            try w.print("heap {d}       {d} MiB{s}\n", .{ i, h.size >> 20, if (h.flags.device_local) " device-local" else "" });
        }
        const exts = try self.vki.enumerateDeviceExtensionPropertiesAlloc(self.physical, null, gpa);
        defer gpa.free(exts);
        const wanted = [_][]const u8{ "VK_KHR_cooperative_matrix", "VK_KHR_shader_bfloat16", "VK_EXT_shader_float8", "VK_KHR_pipeline_executable_properties", "VK_EXT_subgroup_size_control", "VK_KHR_shader_float16_int8", "VK_EXT_shader_atomic_float", "VK_EXT_memory_budget", "VK_KHR_timeline_semaphore", "VK_KHR_calibrated_timestamps", "VK_EXT_calibrated_timestamps" };
        for (wanted) |name| {
            const have = for (exts) |e| {
                if (std.mem.eql(u8, std.mem.sliceTo(&e.extension_name, 0), name)) break true;
            } else false;
            try w.print("extension    {s}: {s}\n", .{ name, if (have) "yes" else "no" });
        }
        try w.print("extensions   {d} in all\n", .{exts.len});
    }

    /// Index of the first memory type allowed by `type_bits` that has every `required` flag.
    /// Among those, one that also has every `preferred` flag wins.
    pub fn memoryType(self: *const Context, type_bits: u32, required: vk.MemoryPropertyFlags, preferred: vk.MemoryPropertyFlags) !u32 {
        const mem = self.vki.getPhysicalDeviceMemoryProperties(self.physical);
        const want = required.merge(preferred);
        var fallback: ?u32 = null;
        for (mem.memory_types[0..mem.memory_type_count], 0..) |t, i| {
            if (type_bits & (@as(u32, 1) << @intCast(i)) == 0) continue;
            if (!t.property_flags.contains(required)) continue;
            if (t.property_flags.contains(want)) return @intCast(i);
            if (fallback == null) fallback = @intCast(i);
        }
        return fallback orelse error.NoSuitableMemoryType;
    }
};

/// A buffer with its own memory allocation. Host-visible buffers stay mapped for their life.
pub const Buffer = struct {
    handle: vk.Buffer,
    memory: vk.DeviceMemory,
    size: vk.DeviceSize,
    /// Non-null when the memory is host-visible.
    mapped: ?[*]u8,

    pub fn init(
        ctx: *const Context,
        size: vk.DeviceSize,
        usage: vk.BufferUsageFlags,
        required: vk.MemoryPropertyFlags,
        preferred: vk.MemoryPropertyFlags,
    ) !Buffer {
        const vkd = ctx.vkd;
        const handle = try vkd.createBuffer(ctx.device, &.{
            .size = size,
            .usage = usage,
            .sharing_mode = .exclusive,
        }, null);
        errdefer vkd.destroyBuffer(ctx.device, handle, null);

        const req = vkd.getBufferMemoryRequirements(ctx.device, handle);
        const memory = try vkd.allocateMemory(ctx.device, &.{
            .allocation_size = req.size,
            .memory_type_index = try ctx.memoryType(req.memory_type_bits, required, preferred),
        }, null);
        errdefer vkd.freeMemory(ctx.device, memory, null);
        try vkd.bindBufferMemory(ctx.device, handle, memory, 0);

        const mapped: ?[*]u8 = if (required.host_visible)
            @ptrCast(try vkd.mapMemory(ctx.device, memory, 0, vk.WHOLE_SIZE, .{}))
        else
            null;
        return .{ .handle = handle, .memory = memory, .size = size, .mapped = mapped };
    }

    pub fn deinit(self: *Buffer, ctx: *const Context) void {
        if (self.mapped != null) ctx.vkd.unmapMemory(ctx.device, self.memory);
        ctx.vkd.destroyBuffer(ctx.device, self.handle, null);
        ctx.vkd.freeMemory(ctx.device, self.memory, null);
        self.* = undefined;
    }

    /// The mapped memory viewed as `T`s. Only for host-visible buffers.
    pub fn slice(self: *const Buffer, comptime T: type) []T {
        const p: [*]T = @ptrCast(@alignCast(self.mapped.?));
        return p[0 .. self.size / @sizeOf(T)];
    }
};

/// Copy `bytes` into the start of device-local `dst` through a temporary staging buffer, and
/// wait for it. For data that goes to the GPU once and stays there.
pub fn uploadOnce(ctx: *const Context, cmds: *const Commands, dst: vk.Buffer, bytes: []const u8) !void {
    var staging = try Buffer.init(ctx, bytes.len, .{ .transfer_src = true }, .{ .host_visible = true, .host_coherent = true }, .{});
    defer staging.deinit(ctx);
    @memcpy(staging.mapped.?[0..bytes.len], bytes);
    try cmds.begin(ctx);
    ctx.vkd.cmdCopyBuffer(cmds.cmd, staging.handle, dst, &.{.{ .src_offset = 0, .dst_offset = 0, .size = bytes.len }});
    try cmds.submitAndWait(ctx);
}

/// A compute pipeline, its layout, and `set_count` descriptor sets (all with the same
/// bindings 0..n-1 in set 0 of the shader). Several sets let one pipeline switch between
/// resources inside a single command buffer, e.g. ping-pong between two images.
pub const ComputePipeline = struct {
    set_layout: vk.DescriptorSetLayout,
    layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,
    pool: vk.DescriptorPool,
    sets: [max_sets]vk.DescriptorSet,
    set_count: u32,
    kinds: [max_bindings]vk.DescriptorType,
    binding_count: u32,
    push_size: u32,

    pub const max_bindings = 16;
    pub const max_sets = 4;
    pub const max_spec = 16;

    /// Shorthand: `binding_count` storage buffers, one set.
    /// `spirv` must be 4-byte aligned: embed it as
    /// `const spv align(@alignOf(u32)) = @embedFile("x.spv").*;`.
    pub fn init(ctx: *const Context, spirv: []align(4) const u8, binding_count: u32, push_size: u32) !ComputePipeline {
        const all_buffers: [max_bindings]vk.DescriptorType = @splat(.storage_buffer);
        return initWith(ctx, spirv, all_buffers[0..binding_count], push_size, 1);
    }

    pub const Extra = struct {
        /// Pin the subgroup width (e.g. 64) and require full subgroups, for shaders that
        /// depend on how many subgroups a workgroup has. 0 leaves it to the driver.
        subgroup_size: u32 = 0,
        /// Specialization constants: `layout(constant_id = i)` gets spec[i] (32 bits each).
        spec: []const u32 = &.{},
        /// Keep the driver's compile statistics (registers, spills, occupancy) for
        /// `statistic`. Needs VK_KHR_pipeline_executable_properties and its feature.
        capture_statistics: bool = false,
    };

    pub fn initWith(ctx: *const Context, spirv: []align(4) const u8, kinds: []const vk.DescriptorType, push_size: u32, set_count: u32) !ComputePipeline {
        return initExtra(ctx, spirv, kinds, push_size, set_count, .{});
    }

    pub fn initExtra(ctx: *const Context, spirv: []align(4) const u8, kinds: []const vk.DescriptorType, push_size: u32, set_count: u32, extra: Extra) !ComputePipeline {
        std.debug.assert(kinds.len <= max_bindings and set_count >= 1 and set_count <= max_sets);
        const vkd = ctx.vkd;
        const dev = ctx.device;

        const module = try vkd.createShaderModule(dev, &.{
            .code_size = spirv.len,
            .p_code = @ptrCast(spirv.ptr),
        }, null);
        defer vkd.destroyShaderModule(dev, module, null);

        var bindings: [max_bindings]vk.DescriptorSetLayoutBinding = undefined;
        for (bindings[0..kinds.len], kinds, 0..) |*b, kind, i| b.* = .{
            .binding = @intCast(i),
            .descriptor_type = kind,
            .descriptor_count = 1,
            .stage_flags = .{ .compute = true },
        };
        const set_layout = try vkd.createDescriptorSetLayout(dev, &.{
            .binding_count = @intCast(kinds.len),
            .p_bindings = &bindings,
        }, null);
        errdefer vkd.destroyDescriptorSetLayout(dev, set_layout, null);

        const push_range = [_]vk.PushConstantRange{.{ .stage_flags = .{ .compute = true }, .offset = 0, .size = push_size }};
        const layout = try vkd.createPipelineLayout(dev, &.{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&set_layout),
            .push_constant_range_count = if (push_size > 0) 1 else 0,
            .p_push_constant_ranges = &push_range,
        }, null);
        errdefer vkd.destroyPipelineLayout(dev, layout, null);

        const sg_info: vk.PipelineShaderStageRequiredSubgroupSizeCreateInfo = .{ .required_subgroup_size = extra.subgroup_size };
        std.debug.assert(extra.spec.len <= max_spec);
        var entries: [max_spec]vk.SpecializationMapEntry = undefined;
        for (entries[0..extra.spec.len], 0..) |*e, i| e.* = .{ .constant_id = @intCast(i), .offset = @intCast(i * 4), .size = 4 };
        const spec_info: vk.SpecializationInfo = .{
            .map_entry_count = @intCast(extra.spec.len),
            .p_map_entries = &entries,
            .data_size = extra.spec.len * 4,
            .p_data = extra.spec.ptr,
        };
        var pipeline: vk.Pipeline = undefined;
        _ = try vkd.createComputePipelines(dev, .null, &.{.{
            .flags = .{ .capture_statistics_khr = extra.capture_statistics },
            .stage = .{
                .p_next = if (extra.subgroup_size != 0) &sg_info else null,
                .flags = .{ .require_full_subgroups = extra.subgroup_size != 0 },
                .stage = .{ .compute = true },
                .module = module,
                .p_name = "main",
                .p_specialization_info = if (extra.spec.len > 0) &spec_info else null,
            },
            .layout = layout,
            .base_pipeline_index = -1,
        }}, null, @as(*[1]vk.Pipeline, &pipeline));
        errdefer vkd.destroyPipeline(dev, pipeline, null);

        // One pool entry per descriptor type used, sized for every set.
        var sizes: [max_bindings]vk.DescriptorPoolSize = undefined;
        var size_count: u32 = 0;
        for (kinds) |kind| {
            for (sizes[0..size_count]) |*ps| {
                if (ps.type == kind) {
                    ps.descriptor_count += set_count;
                    break;
                }
            } else {
                sizes[size_count] = .{ .type = kind, .descriptor_count = set_count };
                size_count += 1;
            }
        }
        const pool = try vkd.createDescriptorPool(dev, &.{
            .max_sets = set_count,
            .pool_size_count = size_count,
            .p_pool_sizes = &sizes,
        }, null);
        errdefer vkd.destroyDescriptorPool(dev, pool, null);

        var layouts: [max_sets]vk.DescriptorSetLayout = undefined;
        @memset(layouts[0..set_count], set_layout);
        var sets: [max_sets]vk.DescriptorSet = undefined;
        try vkd.allocateDescriptorSets(dev, &.{
            .descriptor_pool = pool,
            .descriptor_set_count = set_count,
            .p_set_layouts = &layouts,
        }, &sets);

        var kind_copy: [max_bindings]vk.DescriptorType = undefined;
        @memcpy(kind_copy[0..kinds.len], kinds);
        return .{
            .set_layout = set_layout,
            .layout = layout,
            .pipeline = pipeline,
            .pool = pool,
            .sets = sets,
            .set_count = set_count,
            .kinds = kind_copy,
            .binding_count = @intCast(kinds.len),
            .push_size = push_size,
        };
    }

    pub fn deinit(self: *ComputePipeline, ctx: *const Context) void {
        const vkd = ctx.vkd;
        vkd.destroyDescriptorPool(ctx.device, self.pool, null); // frees the sets too
        vkd.destroyPipeline(ctx.device, self.pipeline, null);
        vkd.destroyPipelineLayout(ctx.device, self.layout, null);
        vkd.destroyDescriptorSetLayout(ctx.device, self.set_layout, null);
        self.* = undefined;
    }

    /// One compile statistic of the pipeline's (only) executable, by the driver's name for it
    /// (RADV: "VGPRs", "Spilled VGPRs", "Subgroups per SIMD", ...). Integers and booleans only;
    /// null if the driver has no such statistic. Needs `capture_statistics` at creation.
    pub fn statistic(self: *const ComputePipeline, ctx: *const Context, name: []const u8) !?u64 {
        var stats: [64]vk.PipelineExecutableStatisticKHR = undefined;
        for (try self.statistics(ctx, &stats)) |st| {
            if (!std.mem.eql(u8, std.mem.sliceTo(&st.name, 0), name)) continue;
            return switch (st.format) {
                .bool32_khr => @intFromBool(st.value.b_32 == .true),
                .int64_khr => @intCast(@max(st.value.i_64, 0)),
                .uint64_khr => st.value.u_64,
                else => null,
            };
        }
        return null;
    }

    /// All compile statistics of executable 0, written into `buf`; returns the filled part.
    pub fn statistics(self: *const ComputePipeline, ctx: *const Context, buf: []vk.PipelineExecutableStatisticKHR) ![]vk.PipelineExecutableStatisticKHR {
        var count: u32 = @intCast(buf.len);
        for (buf) |*st| st.* = .{ .name = undefined, .description = undefined, .format = .bool32_khr, .value = .{ .u_64 = 0 } };
        _ = try ctx.vkd.getPipelineExecutableStatisticsKHR(ctx.device, &.{ .pipeline = self.pipeline, .executable_index = 0 }, &count, buf.ptr);
        return buf[0..count];
    }

    /// Point binding i of set 0 at buffers[i] (whole buffer).
    pub fn bindBuffers(self: *const ComputePipeline, ctx: *const Context, buffers: []const vk.Buffer) void {
        self.bindBuffersSet(ctx, 0, buffers);
    }

    /// Point binding i of set `set` at buffers[i] (whole buffer).
    pub fn bindBuffersSet(self: *const ComputePipeline, ctx: *const Context, set: u32, buffers: []const vk.Buffer) void {
        std.debug.assert(buffers.len == self.binding_count and set < self.set_count);
        var infos: [max_bindings]vk.DescriptorBufferInfo = undefined;
        var writes: [max_bindings]vk.WriteDescriptorSet = undefined;
        for (buffers, 0..) |buf, i| {
            std.debug.assert(self.kinds[i] == .storage_buffer);
            infos[i] = .{ .buffer = buf, .offset = 0, .range = vk.WHOLE_SIZE };
            writes[i] = .{
                .dst_set = self.sets[set],
                .dst_binding = @intCast(i),
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_image_info = undefined,
                .p_buffer_info = @ptrCast(&infos[i]),
                .p_texel_buffer_view = undefined,
            };
        }
        ctx.vkd.updateDescriptorSets(ctx.device, writes[0..buffers.len], null);
    }

    /// Point binding i of set `set` at views[i], a storage image in GENERAL layout.
    pub fn bindImages(self: *const ComputePipeline, ctx: *const Context, set: u32, views: []const vk.ImageView) void {
        std.debug.assert(views.len == self.binding_count and set < self.set_count);
        var infos: [max_bindings]vk.DescriptorImageInfo = undefined;
        var writes: [max_bindings]vk.WriteDescriptorSet = undefined;
        for (views, 0..) |view, i| {
            std.debug.assert(self.kinds[i] == .storage_image);
            infos[i] = .{ .sampler = .null, .image_view = view, .image_layout = .general };
            writes[i] = .{
                .dst_set = self.sets[set],
                .dst_binding = @intCast(i),
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_image,
                .p_image_info = @ptrCast(&infos[i]),
                .p_buffer_info = undefined,
                .p_texel_buffer_view = undefined,
            };
        }
        ctx.vkd.updateDescriptorSets(ctx.device, writes[0..views.len], null);
    }

    /// Record: bind pipeline + set 0, push constants, dispatch.
    pub fn dispatch(self: *const ComputePipeline, ctx: *const Context, cmd: vk.CommandBuffer, push: ?*const anyopaque, groups: [3]u32) void {
        self.dispatchSet(ctx, cmd, 0, push, groups);
    }

    pub fn dispatchSet(self: *const ComputePipeline, ctx: *const Context, cmd: vk.CommandBuffer, set: u32, push: ?*const anyopaque, groups: [3]u32) void {
        const vkd = ctx.vkd;
        vkd.cmdBindPipeline(cmd, .compute, self.pipeline);
        vkd.cmdBindDescriptorSets(cmd, .compute, self.layout, 0, &.{self.sets[set]}, null);
        if (push) |p| vkd.cmdPushConstants(cmd, self.layout, .{ .compute = true }, 0, self.push_size, p);
        vkd.cmdDispatch(cmd, groups[0], groups[1], groups[2]);
    }
};

/// A 2D image with its own memory and a view of the whole image. Starts in UNDEFINED layout;
/// `transition` moves it between layouts.
pub const Image = struct {
    handle: vk.Image,
    memory: vk.DeviceMemory,
    view: vk.ImageView,
    format: vk.Format,
    width: u32,
    height: u32,

    pub fn init(ctx: *const Context, width: u32, height: u32, format: vk.Format, usage: vk.ImageUsageFlags) !Image {
        const vkd = ctx.vkd;
        const handle = try vkd.createImage(ctx.device, &.{
            .image_type = .@"2d",
            .format = format,
            .extent = .{ .width = width, .height = height, .depth = 1 },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1" = true },
            .tiling = .optimal,
            .usage = usage,
            .sharing_mode = .exclusive,
            .initial_layout = .undefined,
        }, null);
        errdefer vkd.destroyImage(ctx.device, handle, null);
        const req = vkd.getImageMemoryRequirements(ctx.device, handle);
        const memory = try vkd.allocateMemory(ctx.device, &.{
            .allocation_size = req.size,
            .memory_type_index = try ctx.memoryType(req.memory_type_bits, .{ .device_local = true }, .{}),
        }, null);
        errdefer vkd.freeMemory(ctx.device, memory, null);
        try vkd.bindImageMemory(ctx.device, handle, memory, 0);
        const view = try vkd.createImageView(ctx.device, &.{
            .image = handle,
            .view_type = .@"2d",
            .format = format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = color_range,
        }, null);
        return .{ .handle = handle, .memory = memory, .view = view, .format = format, .width = width, .height = height };
    }

    pub fn deinit(self: *Image, ctx: *const Context) void {
        ctx.vkd.destroyImageView(ctx.device, self.view, null);
        ctx.vkd.destroyImage(ctx.device, self.handle, null);
        ctx.vkd.freeMemory(ctx.device, self.memory, null);
        self.* = undefined;
    }

    pub const color_range: vk.ImageSubresourceRange = .{
        .aspect_mask = .{ .color = true },
        .base_mip_level = 0,
        .level_count = 1,
        .base_array_layer = 0,
        .layer_count = 1,
    };

    /// Record a layout change, ordered after `src` stages' accesses and before `dst` stages'.
    pub fn transition(
        self: *const Image,
        ctx: *const Context,
        cmd: vk.CommandBuffer,
        old: vk.ImageLayout,
        new: vk.ImageLayout,
        src: vk.PipelineStageFlags2,
        src_access: vk.AccessFlags2,
        dst: vk.PipelineStageFlags2,
        dst_access: vk.AccessFlags2,
    ) void {
        const ib = [_]vk.ImageMemoryBarrier2{.{
            .src_stage_mask = src,
            .src_access_mask = src_access,
            .dst_stage_mask = dst,
            .dst_access_mask = dst_access,
            .old_layout = old,
            .new_layout = new,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = self.handle,
            .subresource_range = color_range,
        }};
        ctx.vkd.cmdPipelineBarrier2(cmd, &.{ .image_memory_barrier_count = ib.len, .p_image_memory_barriers = &ib });
    }

    /// Record a copy of the whole image from/to a tightly packed buffer.
    pub fn copyFromBuffer(self: *const Image, ctx: *const Context, cmd: vk.CommandBuffer, src: vk.Buffer) void {
        ctx.vkd.cmdCopyBufferToImage(cmd, src, self.handle, .transfer_dst_optimal, &.{self.wholeCopy()});
    }

    pub fn copyToBuffer(self: *const Image, ctx: *const Context, cmd: vk.CommandBuffer, dst: vk.Buffer) void {
        ctx.vkd.cmdCopyImageToBuffer(cmd, self.handle, .transfer_src_optimal, dst, &.{self.wholeCopy()});
    }

    fn wholeCopy(self: *const Image) vk.BufferImageCopy {
        return .{
            .buffer_offset = 0,
            .buffer_row_length = 0,
            .buffer_image_height = 0,
            .image_subresource = .{ .aspect_mask = .{ .color = true }, .mip_level = 0, .base_array_layer = 0, .layer_count = 1 },
            .image_offset = .{ .x = 0, .y = 0, .z = 0 },
            .image_extent = .{ .width = self.width, .height = self.height, .depth = 1 },
        };
    }
};

/// One reusable command buffer plus the fence that says when the GPU has finished it.
pub const Commands = struct {
    pool: vk.CommandPool,
    cmd: vk.CommandBuffer,
    fence: vk.Fence,

    pub fn init(ctx: *const Context) !Commands {
        const vkd = ctx.vkd;
        const pool = try vkd.createCommandPool(ctx.device, &.{
            .flags = .{ .reset_command_buffer = true },
            .queue_family_index = ctx.queue_family,
        }, null);
        errdefer vkd.destroyCommandPool(ctx.device, pool, null);
        var cmd: vk.CommandBuffer = undefined;
        try vkd.allocateCommandBuffers(ctx.device, &.{
            .command_pool = pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, @ptrCast(&cmd));
        const fence = try vkd.createFence(ctx.device, &.{}, null);
        return .{ .pool = pool, .cmd = cmd, .fence = fence };
    }

    pub fn deinit(self: *Commands, ctx: *const Context) void {
        ctx.vkd.destroyFence(ctx.device, self.fence, null);
        ctx.vkd.destroyCommandPool(ctx.device, self.pool, null); // frees the command buffer
        self.* = undefined;
    }

    pub fn begin(self: *const Commands, ctx: *const Context) !void {
        try ctx.vkd.resetCommandBuffer(self.cmd, .{});
        try ctx.vkd.beginCommandBuffer(self.cmd, &.{ .flags = .{ .one_time_submit = true } });
    }

    /// Start recording a buffer that will be submitted many times (`resubmit`).
    pub fn beginReusable(self: *const Commands, ctx: *const Context) !void {
        try ctx.vkd.resetCommandBuffer(self.cmd, .{});
        try ctx.vkd.beginCommandBuffer(self.cmd, &.{});
    }

    /// End recording, submit, and block until the GPU is done.
    pub fn submitAndWait(self: *const Commands, ctx: *const Context) !void {
        try ctx.vkd.endCommandBuffer(self.cmd);
        try self.resubmit(ctx, .block);
    }

    pub const Wait = enum { block, spin, hybrid };
    const spin_limit_ns = 10 * std.time.ns_per_s;
    /// `hybrid` polls this long, then sleeps: short jobs finish while it polls (no wake-up
    /// cost), long ones stop burning a core.
    pub const hybrid_spin_ns = 50 * std.time.ns_per_us;

    /// Submit the already-recorded buffer and wait for it: `block` sleeps in vkWaitForFences,
    /// `spin` polls vkGetFenceStatus (no kernel wake-up, but a busy CPU), `hybrid` polls for
    /// `hybrid_spin_ns`, then blocks.
    pub fn resubmit(self: *const Commands, ctx: *const Context, mode: Wait) !void {
        try self.submit(ctx);
        try self.wait(ctx, mode);
    }

    /// Hand the recorded buffer to the queue and return; `wait` must follow before the next
    /// submit (the fence is reset there).
    pub fn submit(self: *const Commands, ctx: *const Context) !void {
        const cmd_info = [_]vk.CommandBufferSubmitInfo{.{ .command_buffer = self.cmd, .device_mask = 0 }};
        try ctx.vkd.queueSubmit2(ctx.queue, &.{.{
            .command_buffer_info_count = cmd_info.len,
            .p_command_buffer_infos = &cmd_info,
        }}, self.fence);
    }

    /// Wait for the last `submit` to finish (see `resubmit` for the modes), then reset the fence.
    pub fn wait(self: *const Commands, ctx: *const Context, mode: Wait) !void {
        const vkd = ctx.vkd;
        switch (mode) {
            .block => _ = try vkd.waitForFences(ctx.device, &.{self.fence}, .true, std.math.maxInt(u64)),
            .spin => {
                // Bounded like the blocking wait would be by the driver's own timeout: a hung GPU
                // must not spin a core forever.
                const deadline = host.nowNs() + spin_limit_ns;
                while (try vkd.getFenceStatus(ctx.device, self.fence) == .not_ready) {
                    if (host.nowNs() > deadline) return error.FenceSpinTimeout;
                    std.atomic.spinLoopHint();
                }
            },
            .hybrid => {
                const until = host.nowNs() + hybrid_spin_ns;
                while (try vkd.getFenceStatus(ctx.device, self.fence) == .not_ready) {
                    if (host.nowNs() > until) {
                        _ = try vkd.waitForFences(ctx.device, &.{self.fence}, .true, std.math.maxInt(u64));
                        break;
                    }
                    std.atomic.spinLoopHint();
                }
            },
        }
        try vkd.resetFences(ctx.device, &.{self.fence});
    }
};

/// Record a barrier: everything `src` stages wrote must be visible to `dst` stages' reads.
pub fn barrier(ctx: *const Context, cmd: vk.CommandBuffer, src: vk.PipelineStageFlags2, src_access: vk.AccessFlags2, dst: vk.PipelineStageFlags2, dst_access: vk.AccessFlags2) void {
    const mb = [_]vk.MemoryBarrier2{.{
        .src_stage_mask = src,
        .src_access_mask = src_access,
        .dst_stage_mask = dst,
        .dst_access_mask = dst_access,
    }};
    ctx.vkd.cmdPipelineBarrier2(cmd, &.{ .memory_barrier_count = mb.len, .p_memory_barriers = &mb });
}

/// GPU-side clock: timestamps written by the GPU itself as it reaches each point.
pub const Timestamps = struct {
    pool: vk.QueryPool,
    count: u32,
    period_ns: f64,

    pub fn init(ctx: *const Context, count: u32) !Timestamps {
        const pool = try ctx.vkd.createQueryPool(ctx.device, &.{ .query_type = .timestamp, .query_count = count }, null);
        return .{ .pool = pool, .count = count, .period_ns = ctx.props.limits.timestamp_period };
    }

    pub fn deinit(self: *Timestamps, ctx: *const Context) void {
        ctx.vkd.destroyQueryPool(ctx.device, self.pool, null);
        self.* = undefined;
    }

    /// Record at the start of a command buffer, before any `write`.
    pub fn reset(self: *const Timestamps, ctx: *const Context, cmd: vk.CommandBuffer) void {
        ctx.vkd.cmdResetQueryPool(cmd, self.pool, 0, self.count);
    }

    pub fn write(self: *const Timestamps, ctx: *const Context, cmd: vk.CommandBuffer, index: u32) void {
        ctx.vkd.cmdWriteTimestamp2(cmd, .{ .all_commands = true }, self.pool, index);
    }

    /// The raw GPU tick count of timestamp `index`. Call after the submit has finished.
    pub fn tick(self: *const Timestamps, ctx: *const Context, index: u32) !u64 {
        var t: u64 = 0;
        _ = try ctx.vkd.getQueryPoolResults(ctx.device, self.pool, index, 1, @sizeOf(u64), &t, @sizeOf(u64), .{ .@"64" = true, .wait = true });
        return t;
    }

    /// Nanoseconds between timestamp `from` and `to`. Call after the submit has finished.
    pub fn elapsedNs(self: *const Timestamps, ctx: *const Context, from: u32, to: u32) !f64 {
        var ticks: [2]u64 = undefined;
        _ = try ctx.vkd.getQueryPoolResults(ctx.device, self.pool, from, 1, @sizeOf(u64), &ticks[0], @sizeOf(u64), .{ .@"64" = true, .wait = true });
        _ = try ctx.vkd.getQueryPoolResults(ctx.device, self.pool, to, 1, @sizeOf(u64), &ticks[1], @sizeOf(u64), .{ .@"64" = true, .wait = true });
        return @as(f64, @floatFromInt(ticks[1] -% ticks[0])) * self.period_ns;
    }
};

/// A GPU tick and the CPU clock (`host.nowNs`) read together by the driver
/// (VK_KHR_calibrated_timestamps), so GPU timestamps can be placed on the CPU's timeline. The
/// two clocks drift apart by parts per million, so take a fresh one next to each measurement.
pub const Calibration = struct {
    gpu_ticks: u64,
    cpu_ns: u64,
    /// The driver's bound on how far apart the two reads were, in ns.
    deviation_ns: u64,
    period_ns: f64,

    pub const extension = vk.device_extensions.khr_calibrated_timestamps.name;
    /// The CPU clock `host.nowNs` reads: QueryPerformanceCounter on Windows, CLOCK_MONOTONIC on
    /// Linux.
    const cpu_domain: vk.TimeDomainKHR = if (builtin.os.tag == .windows) .query_performance_counter_khr else .clock_monotonic_khr;

    /// Whether this device can calibrate against `host.nowNs`: the extension is on, both
    /// domains are offered, and timestamps do not wrap.
    pub fn available(ctx: *const Context, gpa: std.mem.Allocator) !bool {
        if (!ctx.enabled(extension) or ctx.timestamp_bits < 64) return false;
        var count: u32 = 0;
        _ = try ctx.vki.getPhysicalDeviceCalibrateableTimeDomainsKHR(ctx.physical, &count, null);
        const domains = try gpa.alloc(vk.TimeDomainKHR, count);
        defer gpa.free(domains);
        _ = try ctx.vki.getPhysicalDeviceCalibrateableTimeDomainsKHR(ctx.physical, &count, domains.ptr);
        var device = false;
        var cpu = false;
        for (domains[0..count]) |d| {
            if (d == .device_khr) device = true;
            if (d == cpu_domain) cpu = true;
        }
        return device and cpu;
    }

    pub fn take(ctx: *const Context) !Calibration {
        const infos = [_]vk.CalibratedTimestampInfoKHR{ .{ .time_domain = .device_khr }, .{ .time_domain = cpu_domain } };
        var stamps: [2]u64 = undefined;
        const deviation = try ctx.vkd.getCalibratedTimestampsKHR(ctx.device, &infos, &stamps);
        return .{ .gpu_ticks = stamps[0], .cpu_ns = host.cpuDomainNs(stamps[1]), .deviation_ns = deviation, .period_ns = ctx.props.limits.timestamp_period };
    }

    /// GPU tick `t` on the CPU clock, in ns relative to CPU time `base` (so the f64 keeps
    /// sub-ns precision; absolute ns since boot would lose it after a few months).
    pub fn cpuNsOf(self: Calibration, t: u64, base: u64) f64 {
        // Signed: a tick taken before the calibration is behind it. Wrapping subtraction is
        // exact because `available` requires 64-bit timestamps.
        const dt: i64 = @bitCast(t -% self.gpu_ticks);
        const from_base: i64 = @bitCast(self.cpu_ns -% base);
        return @as(f64, @floatFromInt(from_base)) + @as(f64, @floatFromInt(dt)) * self.period_ns;
    }
};

fn hasLayer(gpa: std.mem.Allocator, vkb: vk.BaseWrapper, name: [*:0]const u8) !bool {
    const layers = try vkb.enumerateInstanceLayerPropertiesAlloc(gpa);
    defer gpa.free(layers);
    for (layers) |l| {
        if (std.mem.eql(u8, std.mem.sliceTo(&l.layer_name, 0), std.mem.span(name))) return true;
    }
    return false;
}

fn pickDevice(vki: vk.InstanceWrapper, physicals: []const vk.PhysicalDevice, index: ?usize) !vk.PhysicalDevice {
    if (physicals.len == 0) return error.NoVulkanDevice;
    if (index) |i| {
        if (i >= physicals.len) return error.DeviceIndexOutOfRange;
        return physicals[i];
    }
    for (physicals) |p| {
        if (vki.getPhysicalDeviceProperties(p).device_type == .discrete_gpu) return p;
    }
    return physicals[0];
}

const Family = struct { index: u32, timestamp_bits: u32 };

fn computeQueueFamily(gpa: std.mem.Allocator, vki: vk.InstanceWrapper, physical: vk.PhysicalDevice) !Family {
    const families = try vki.getPhysicalDeviceQueueFamilyPropertiesAlloc(physical, gpa);
    defer gpa.free(families);
    for (families, 0..) |f, i| {
        if (f.queue_flags.compute) return .{ .index = @intCast(i), .timestamp_bits = f.timestamp_valid_bits };
    }
    return error.NoComputeQueue;
}

fn debugCallback(
    severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    _: vk.DebugUtilsMessageTypeFlagsEXT,
    data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    _: ?*anyopaque,
) callconv(vk.vulkan_call_conv) vk.Bool32 {
    const level = if (severity.error_ext) "error" else "warning";
    const msg = if (data) |d| d.p_message orelse "(no message)" else "(no data)";
    // The loader reports the host's layer setup (on Windows: stale registry entries from other
    // apps, duplicate driver manifests, layer files this account may not read) under this id.
    // Not a fault of this program, so not counted, and shown only when asked
    // (`show_loader_notes`; never in tests, whose runner fails on any stderr output).
    const id: []const u8 = if (data) |d| if (d.p_message_id_name) |n| std.mem.span(n) else "" else "";
    if (std.mem.eql(u8, id, "Loader Message")) {
        if (show_loader_notes) report.note("vulkan loader {s}: {s}\n", .{ level, msg });
        return .false;
    }
    std.debug.print("vulkan {s}: {s}\n", .{ level, msg });
    _ = validation_messages.fetchAdd(1, .monotonic);
    return .false;
}

test "opens the GPU with validation on and no validation messages" {
    report.progress = false;
    var ctx = try Context.init(std.testing.allocator, .{ .validation = true });
    defer ctx.deinit();
    try std.testing.expect(ctx.messenger != .null);
    try std.testing.expectEqual(@as(u32, 0), validation_messages.load(.monotonic));
}
