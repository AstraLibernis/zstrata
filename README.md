# zstrata

A map of this machine's layers, measured with one method on every device, and the same tasks
run on each device to see where each kind of work and data belongs. It is the reference the
other projects are checked against, so it depends on none of them (its thread pool is its own,
`src/cpu/pool.zig`); the only dependencies are the Vulkan bindings and headers.

- **cpu**: the core clock (one core busy, all busy), vector width (scalar → AVX-512) ×
  independent chains, threads and SMT, cache and RAM
  latency/bandwidth, branch prediction, cost of handing work to threads (spawn with Zig's
  default and a 1 MiB stack, pool round trips hot and after idle), first-touch memory
- **link**: CPU ↔ GPU: staged vs direct (resizable BAR) uploads, downloads, the fixed cost of
  one GPU round trip and where it goes (recording, waiting: block / spin / hybrid), what each
  wait costs the CPU on a long job, batching many dispatches into one submit, and one trip
  split into phases on the CPU's clock (record, submit, launch, GPU run, wake; GPU timestamps
  placed with VK_KHR_calibrated_timestamps), also after the machine sat idle 1 ms to 1 s
- **check**: every full run takes three markers (L1 latency, RAM latency, FMA on one thread per
  core) before each part and at the end, and flags itself (`cpu,check,disturbed`) when one moves
  more than 10%: something else was using the machine. Each checkpoint also records the clock
  (`cpu,check,clock`), so a marker that moved with the clock reads as boost, not code. `zig
  build spread` also compares each run's markers with the other runs'
- **gpu**: registers and wave size, busy lanes, shared memory, caches and VRAM, matrix units,
  networks by width, with weights read one float or four at a time (first built in an earlier Vulkan lab project)
- **cross**: the same task on 1 core, all threads, and the GPU (kernel only, with the round
  trip, with transfers): streaming math, neural networks, a decision-tree ensemble, step-by-step
  chains

```sh
zig build run -Doptimize=ReleaseFast -- all > docs/data/strata.csv   # everything, one CSV
zig build run -Doptimize=ReleaseFast -- cpu --only alu,chase         # some probes (no check)
zig build run -Doptimize=ReleaseFast -- all --no-check               # full run without the check
zig build run -Doptimize=ReleaseFast -- all --no-throttle            # opt out of Windows power throttling
zig build test                                                       # Debug: every probe, small sizes
zig-out/bin/zstrata --device-info    # which driver answers, compute limits, heaps, extensions
zig-out/bin/zstrata gpu --list-stats # the per-shader statistics this driver reports
```

- Plan and progress: [`docs/PLAN.md`](docs/PLAN.md)
- Numbers and what they mean: [`docs/measurements.md`](docs/measurements.md)
- What to do in other projects to stay fast (measured, both OSes):
  [`docs/speed-playbook.md`](docs/speed-playbook.md)
- Page: `docs/page/strata.html` (rebuild: `zig build page`)
- Run-to-run spread: `zig build spread -- docs/data/runs/run*.csv`

Runs on Linux and Windows from the same source; OS-specific calls sit behind
`src/cpu/host.zig` (clock, pinning, sleep, topology, thread CPU time), `src/cpu/pool.zig`
(sleeping workers) and the loader in `src/gpu.zig`.

- **Linux:** Vulkan through RADV, using the persisted stack in `~/.local/vulkan`
  (`tools/env.sh`; `VKPREFIX` overrides).
- **Windows:** AMD's driver (it ships `vulkan-1.dll`). The build needs `glslc` from shaderc at
  `%USERPROFILE%\.local\shaderc\bin` (`VKPREFIX` overrides); the tests need the Khronos
  validation layer at `%USERPROFILE%\.local\vk-layers` (`VKLAYERS` overrides). Both are
  per-user downloads (shaderc's CI build, the validation layers' `nightly` release).

`src/gpu.zig` and the GPU probe shaders come from an earlier Vulkan lab project by the same author.

## As a library

Other projects can import the parts that measured fastest (`src/lib.zig`):

- `Pool`: the thread pool (`parallelFor` over a range; `chunkFor` says how the range is cut).
- `simd`: the vector width for the CPU being built for, and a multiply-add that is fused only
  where the CPU has the instruction.
- `sort`: radix sorts for unsigned integers, for floats, and for row indices ordered by a
  float column.

In the other project's `build.zig`:

    const zstrata = b.dependency("zstrata", .{ .target = target, .optimize = optimize, .@"lib-only" = true });
    exe.root_module.addImport("zstrata", zstrata.module("zstrata"));

`lib-only` leaves out the measuring program, so neither the Vulkan packages nor the shader
compiler are needed. `zig build test-lib` runs the library's tests without a GPU.

## Licence

The measuring program is GPL-3.0-or-later (`COPYING`). The library files `src/lib.zig` and
`src/lib/` are LGPL-3.0-or-later (`COPYING.LESSER`); each file's header says which applies.
