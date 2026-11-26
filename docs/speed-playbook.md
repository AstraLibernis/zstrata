# Speed playbook

What to do in the other projects to stay fast on this machine (Ryzen 7 9800X3D, RX 9070 XT), on
Linux (RADV) and Windows 11 (AMD driver). Every rule has a measurement behind it in this repo;
the probe that measures it is named so it can be re-run (`zig build run -Doptimize=ReleaseFast
-- <part> --only <probe>`). Applying a rule to another project happens when that project is
being worked on, one at a time.

Windows numbers: `docs/data/runs/run15-18-windows.csv` (spread
`docs/data/strata-spread-windows-speed.csv`), medians of 4 full runs, 2026-11-08. Linux numbers:
runs 1-5 (2026-10-15); rules 14-18 from native Fedora, runs 29-36 (2026-11-26,
`measurements.md`, "Linux gaps"). Rows marked "Windows only" predate those runs. Tags:
[MEASURED] from these runs, [DERIVED] reasoned from them.

## The rules

### 1. Batch GPU work: one submit for many dispatches
Every submit + wait costs a fixed ~40-50 us on Windows (Linux ~34 us), however small the work.
Dependent dispatches in one command buffer, with a barrier between each, pay it once:

| dispatches | one submit each | all in one submit (per dispatch) |
|---|---|---|
| 1 | 41.3 us | 45.6 us |
| 4 | 42.4 us | 12.5 us |
| 16 | 46.8 us | 4.1 us |
| 64 | 44.4 us | 1.5 us |
| 256 | 47.7 us | 0.85 us |

[MEASURED, `link --only batch`, Windows only.] 55x at 256. **Apply:** record a whole training
step, frame or pipeline of kernels into one command buffer; never submit + wait per layer or
per step. This is the biggest single lever, and it also removes most of the Windows round-trip
penalty (rule 2) because there are fewer round trips to pay for.

### 2. Wait for the GPU with a short poll, then sleep (hybrid)
After a submit, how the CPU waits decides both latency and the CPU it burns:

| wait | empty round trip | 2.9 ms job: wall | 2.9 ms job: CPU used |
|---|---|---|---|
| block (vkWaitForFences) | 49.4 us | 2885 us | 41 us |
| spin (poll vkGetFenceStatus) | 44.2 us | 2861 us | 2883 us (a whole core) |
| hybrid (poll 50 us, then block) | 44.1 us | 2886 us | 91 us |

[MEASURED, `link --only roundtrip,wait`, prerecorded buffer, Windows only.] Blocking costs
~5-10 us on Windows because the kernel has to wake the thread; Linux's blocking round trip is
already 33.8 us. **Apply:** `Commands.resubmit(ctx, .hybrid)` from `src/gpu.zig` (bounded: the
spin has a 10 s guard) wherever a short GPU job sits on the critical path (inference of one
batch, readbacks). Plain `.block` everywhere else. Never `.spin` on long jobs.

### 3. Record once, resubmit
Re-recording an empty command buffer each time costs ~4-5 us of the round trip on Windows
(empty-submit 53.6 vs empty-resubmit 49.4 us; with spin 46.4 vs 44.2). [MEASURED.] **Apply:**
for work that repeats with the same shape, record the command buffer once (no
`one_time_submit`) and resubmit it; change inputs through buffers or push constants.

### 4. Keep data on the GPU between steps
The crossovers on both OSes: streaming math (y = a*x + b) never pays on the GPU when the data
crosses the link each time (64M floats: 34.7 ms with transfer vs 11.4 ms on all CPU threads),
and wins 11x when the data already lives in VRAM (1.0 ms). Networks pay from 64 at a time, trees
from 4K rows, chains from 64K, one long chain never. [MEASURED, `cross`, same verdicts on both
OSes.] **Apply:** upload once, run many steps, download once. A long chain of dependent steps
stays on one CPU core (1.15 ns/step vs 7-16 ns/step on the GPU).

### 5. Load 16 bytes at a time in shaders (vec4)
The GPU compilers do not reliably merge neighbouring float loads. Written as float loads, the
network kernel reading weights from shared memory got `ds_load_2addr_b32` (two floats per
instruction) from AMD's compiler; written as vec4 it gets `ds_load_b128` (four):

| network kernel, weights in | width | floats (TFLOP/s) | vec4 (TFLOP/s) |
|---|---|---|---|
| shared memory, 1 net | W=32 | 0.37 | 1.01 (2.7x) |
| shared memory, 1 net | W=64 | 0.40 | 1.38 (3.4x) |
| shared memory, 8 nets packed | W=8 | 1.36 | 3.24 (2.4x) |
| buffer, 1 net | W=128 | 0.61 | 3.63 (5.9x) |
| buffer, 1 net | W=192 | 0.13 | 0.97 (7.2x) |
| buffer, 1 net | W=256 | 0.12 | 0.40 (3.4x) |
| shared memory, 1 net | W=8 | 0.59 | 0.50 (0.85x, worse) |

[MEASURED, `gpu --only mlp`, variants `lds`/`lds4`, `mem`/`mem4`; Windows only; the vec4
versions do the same fma chain in the same order, so results are bit-equal and checked against
the CPU.] **Apply:** in compute shaders, declare shared arrays and storage buffers as `vec4[]`
where the data is read in runs of 4 (W % 4 == 0), and keep the arithmetic order. Then measure:
it lost once (W=8, one net).

### 6. Check what the compiler did before blaming the OS
Same SPIR-V, different drivers: RADV and AMD's Windows compiler chose opposite register
strategies (network kernel, weights from memory: W=64 48 vs 165 VGPRs, W=256 256 + spills vs 33),
and each won where the other lost (0.12x to 6.7x). [MEASURED.] Both drivers report per-shader
statistics (zstrata's vgprs/sgprs/scratch/lds columns; `gpu --list-stats` shows the names: RADV
`VGPRs`, AMD `numUsedVgprs`), but AMD's driver has no spill or occupancy counts; a non-zero
scratch size means spills. To see the instructions themselves on Windows, use AMD's Radeon GPU
Analyzer, live-driver mode:

```sh
spirv-opt --set-spec-const-default-value "0:64 1:1" in.spv -o w64.spv     # bake spec constants
rga -s vulkan -c gfx1201 --comp w64.spv -a stats.csv --isa w64.isa       # VGPRs, LDS, ISA
```

(`~/.local/rga-2.15`, `spirv-opt` in `~/.local/shaderc/bin`.) Wave64 code uses
`s_..._b64 exec`; wave32 uses `exec_lo`. **Apply:** when a kernel is slow on one OS only, read
its VGPRs and load widths first.

### 7. Prefer registers for small networks, on both OSes
Taking the best placement at each width, the two OSes agree: weights in registers win from
W=8 to W=192, and the best Windows number is within 0.95-1.18x of the best Linux one, except
W=32 (0.84x), W=96 (0.69x) and W=128 (0.78x). [MEASURED, best of every `mlp` variant, Linux runs
1-4 vs Windows runs 15-18.] **Apply:** default to register-resident weights up to W=192; only
past that consider buffers (with vec4).

### 8. Read device limits at run time
The same card reports 64 KiB of shared memory per workgroup on RADV and 32 KiB on AMD's Windows
driver. A pipeline above the limit is invalid (the validation layer reports
VUID-RuntimeSpirv-Workgroup-06530) even if the driver runs it. [MEASURED.] **Apply:** check
`props.limits.max_compute_shared_memory_size` (zstrata: `Lab.sharedLimit()`) before choosing a
tile or network size, and keep the validation layer on in tests (it caught this).

### 9. Reuse threads; if you spawn, pass a stack size
| | Linux | Windows |
|---|---|---|
| spawn + join, Zig default stack (16 MiB) | 9 us | 92 us |
| spawn + join, 1 MiB stack | not run | 58 us |
| hand work to a pool, workers awake | 0.16-0.56 us (2-8 threads) | 0.12-0.49 us |

[MEASURED, `cpu --only sync`.] Zig's std commits the whole default stack up front on Windows
(CreateThread with 16 MiB committed: 108 us; reserved only: 75 us). **Apply:** keep a pool for
any repeated parallel work; spawn with `.stack_size` set to what the thread needs (1 MiB is
plenty for ordinary code); never spawn per task.

### 10. Let idle pool workers sleep, not spin or yield
A pool whose workers yield forever looks fast (0.4 us to start work after idle) but keeps every
core busy. Sleeping on the epoch (futex on Linux, `RtlWaitOnAddress` on Windows) costs 10.5 /
15.9 / 26.0 us to wake 2 / 4 / 8 threads on Windows (Linux 20-29 us). [MEASURED,
`pool-after-idle`.] **Apply:** spin a budget, then park: `src/cpu/pool.zig` is the reference;
its tests assert that idle helpers are really asleep.

### 11. Reuse allocations
Writing freshly allocated memory runs at half the speed of rewriting it: 1 GiB first write 11.3
vs 20.9 GB/s on Windows (Linux 4.6 vs 10.4). [MEASURED, `cpu --only alloc`.] **Apply:** keep
scratch buffers across rounds instead of allocating per round.

### 12. Leave headroom on the CPU when running below normal priority
These runs are at BelowNormal priority (so the desktop stays responsive), so with all 16 logical CPUs
busy any other thread preempts one of ours: 16 threads reach ~0.90x of Linux, 12 threads
0.995x. A 16-thread pool barrier also gets slow and noisy (pool after idle 79-146 us).
[MEASURED, `cpu --only alu,sync`; the cause is DERIVED.] **Apply:** for latency-sensitive
parallel work use up to 12-14 threads; 16 only for long throughput runs.

### 13. Pick the CPU target on purpose: native here, `x86_64_v3` to ship
`zig build` without `-Dcpu` builds for this CPU (znver5, AVX-512); such a binary fails with an
illegal instruction on CPUs without those instructions. What each target costs on this machine
(medians of 3 runs per build, against native):

| target | AVX-512-shaped math | small nets, 1 core / all | L1 reads | RAM-bound work |
|---|---|---|---|---|
| `x86_64_v4` (AVX-512) | 1.00x | 1.0x / 1.0x | 1.0x | 1.0x |
| `x86_64_v3` (AVX2 + FMA) | 0.35x | 1.44x / 1.21x slower | 0.52x | 1.0x |
| `x86_64_v2`, `baseline` (no FMA) | 0.006x | 11-12x slower | 0.10x | 1.0x |

[MEASURED, `cpu cross` per build, Windows.] Zen 5 tuning on top of v4 buys nothing measurable.
v3 loses more than half on code written for 16-wide vectors with many chains: AVX2 has half the
width and 16 vector registers instead of 32, so 16 chains of 16 lanes spill [DERIVED; no
disassembler here]. Below v3 there is no FMA instruction, and `@mulAdd` becomes a software
routine: even scalar math runs 10x slower [DERIVED from scalar 0.0020 vs 0.0197 TFLOP/s].
**Apply:** keep native builds for work on this machine; for anything shipped, build
`-Dcpu=x86_64_v3` (any x86 CPU from about 2015) and size vectors with
`std.simd.suggestVectorLength` rather than a fixed 16. Never ship below v3 code that uses
`@mulAdd`.

### 14. Align big buffers to the page, not to `malloc`
With libc linked, Zig's general allocator is glibc `malloc`, and a big block from it starts 16
bytes into a cache line: every 64-byte vector load then touches two lines. One thread reading
an L1-sized set: 334 GB/s from such a block, 643 page-aligned (Windows' heap already aligns:
633). [MEASURED, `cpu --only stream`.] **Apply:** buffers read or written with vectors come from
`zstrata.mem.allocBig` (`src/lib/mem.zig`), or `alignedAlloc(T, .@"64", n)`.

### 15. On Linux, ask for huge pages
Transparent huge pages are `madvise`-only on Fedora, so a plain allocation gets 4 KiB pages:
1 GiB first write 8.2 GB/s against 24.1 with `MADV_HUGEPAGE` (Windows 10.9); random reads in
64 MiB 17.9 -> 10.8 ns, in 1 GiB 89 -> 75 ns. Rewriting is the same either way (21 GB/s).
[MEASURED, `cpu --only alloc,chase`.] **Apply:** `allocBig(T, n, .huge)` for anything from 2 MiB
up. No privilege needed. Rule 11 still holds, with a smaller factor.

### 16. On Linux, a cold GPU runs at half clock: keep it warm, or make jobs long
amdgpu's default power level starts work at ~1.6 GHz and reaches 3.2-3.4 GHz only after ~0.5 s
of unbroken load; ~0.1 s of light load drops it again. Every job shorter than that pays 2x
(1M-step chain 16.7 ms cold, 8.3 ms warmed; Windows 7.2). `gpu/warm.zig` keeps one lane busy
on the second compute queue and holds the clock, with no root. It costs: staged uploads
0.5-0.7x, launch of a round trip 20 -> 71 us. [MEASURED, `--gpu-warm`, `tools/linux/clkrun.py`.]
**Apply:** for bursts of short kernels (inference, per-frame compute) run a warmer while the
burst lasts, with jobs of ~0.2 s (short warmer jobs flush the GPU caches; one of 4 s stalled
the other queue); not while streaming data over the link. Rule 1 helps here too: one long
submit ramps the clock, many short ones never do.

### 17. In shaders, do several steps per loop round
RADV's loop header (`s_wait_alu`, compare, branch) costs about one multiply, and RADV loads a
buffer's descriptor again at every use, so a tight loop pays both on every step. Unrolled by
hand: dependent float chain 8.3 -> 5.5 ns/step (4 per round; Windows 7.2 with the old shader),
scalar load chain 20.4 -> 11.7 ns (32 per round; Windows 11.7), fma at 4 values per lane 33 ->
42 TFLOP/s (Windows 39). Same operations in the same order, so results stay bit-equal.
[MEASURED, `cross --only chains,chain-long`, `gpu --only chase,alu`; `RADV_DEBUG=shaders`
shows the code.] **Apply:** in hot shader loops with a small body, write 4-32 steps per round
plus a remainder loop; move loop-invariant buffer reads into locals before the loop (RADV does
not, AMD's compiler does: rule 7).

### 18. Measure the GPU with the screen on
With the display off (locked desktop), the GPU's memory clock idles at 96 MHz instead of 1258
and follows load slowly: GPU latency past the caches read 430-850 ns instead of 200-330.
[MEASURED, runs 33-36 against a rerun under the stable pstate.] **Apply:** note the display
state with any GPU memory number; for an unattended run hold the memory clock with
`tools/linux/pstate.py` (which also fixes the core clock at 2370 MHz, so not for ALU rows).

## How to measure, so the numbers hold

- ReleaseFast for timing, Debug/ReleaseSafe for correctness. Never time Debug.
- Judge from full runs (medians of 4), not spot checks. Right after a rebuild a fresh executable
  runs slower at first (spawn read 108-140 us in spot checks vs 89-98 us in every full run;
  Defender scanning new executables is the likely cause [DERIVED]).
- Compare like with like: CPU rows that moved by the same factor at every vector width and
  chain count are the clock, not the code (Windows boosts ~90 MHz lower here, 0.98x everywhere).
- When a shader changes, rebuild the unchanged variants too and `cmp` their SPIR-V: byte-equal
  means the old numbers still stand.

## Where each rule applies (to do when the project is worked on)

| project | rules most likely to pay [DERIVED] |
|---|---|
| GPU network projects | 1, 2, 3, 4, 5, 6, 7, 8 |
| Vulkan lab projects | 1, 2, 5, 6, 8 |
| zmural (GPU wallpaper) | 1, 3, 8; the frame loop is the submit loop |
| zarbor (trees, CPU) | 9 (its pool spawns helpers with the 16 MiB default), 10 (done in c639dbb), 11, 12; 4 says trees go to the GPU only from ~4K rows |
| CPU math projects | 9, 11, 12 |
