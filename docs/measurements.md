# Measurements

All numbers from `zig build run -Doptimize=ReleaseFast -- all` on 2026-10-15
(`docs/data/strata.csv`; its CPU alu and stream rows re-measured after the start-barrier fix
below; CPU and link also in `cpu-run1.csv` / `link-run1.csv` from an earlier
run of the same code, except the branch probe, fixed in between). Ryzen 7 9800X3D (8 cores, 16
threads, 96 MiB L3), RX 9070 XT (RADV, Mesa 26.1.8), in a container on a desktop that was also in
use. Medians of 5 runs. Tags: [MEASURED] from this code, [DERIVED] computed or inferred from
measurements.

## CPU

- **ALU, one core:** scalar 0.021, SSE 0.084, AVX2 0.167, AVX-512 0.335 TFLOP/s at >= 8
  independent chains; 1 chain: 1/8 of that (4-cycle FMA latency, 2 FMA units [DERIVED]). Past 28
  chains of AVX-512 the 32 vector registers run out and speed halves (0.16). [MEASURED]
- **ALU, threads:** 1 -> 8 threads scales 8x (AVX-512 2.67 TFLOP/s); 16 threads 2.67 (SMT adds
  nothing for FMA-bound code); 12 threads 2.00 (4 cores carry two threads, the rest wait). Run 1
  and run 2 agree to 3 digits. [MEASURED]
- **Latency (random 64-byte lines, one thread):** L1 1.0 ns to 48 KiB; L2 2.9 ns to ~512 KiB,
  3.9-5.5 ns to 1 MiB; L3 10-12 ns to 16 MiB, 14-19 ns to 64 MiB (likely TLB misses with 4 KiB
  pages [DERIVED]); RAM 80-88 ns. [MEASURED]
- **Read bandwidth:** working sets of 8-32 KiB vary up to 2x between runs (one thread: 167-334
  GB/s at 8 KiB), so treat the L1 end of the stream curves as rough; from 64 KiB up runs agree
  within ~25%. [MEASURED] One thread: L1/L2 ~230-330 GB/s, L3 ~150, RAM 59-61. All 16 threads: 2.66
  TB/s while each thread's slice fits L1, ~1.9 TB/s in L2, 0.73-1.07 TB/s L3, RAM 58-60. One
  thread already saturates RAM (single-CCD fabric limit [DERIVED, not verified]). [MEASURED]
- **Branch (64 KiB, in L2):** predictable 0.19 ns/item, random 2.35, branchless 0.38: ~4.4 ns
  (~23 cycles at ~5.2 GHz [DERIVED]) per misprediction. [MEASURED]
- **Sync:** thread spawn+join 9.1 us; zarbor pool round trip, warm: 0.14 (2 thr) - 1.04 us (16);
  after 5 ms idle (workers parked): 26-30 us. [MEASURED]
- **Alloc:** writing 1 GiB of fresh memory 4.6 GB/s, the same memory again 10.4 GB/s. [MEASURED]

## Link

- **Upload:** CPU writing VRAM directly (resizable BAR, whole 16 GB mapped) 22-23 GB/s from 256
  KiB up; staged (memcpy + GPU copy) up to 14 GB/s; the GPU copy alone 28 GB/s. [MEASURED]
- **Download:** GPU copy into cached CPU memory + memcpy out: 10.5-13 GB/s from 16 MiB; CPU
  reading VRAM directly 0.07 GB/s at every size. [MEASURED]
- **Round trip:** empty submit + wait 33.8 us; tiny dispatch + 4-byte read-back 36.7 us
  (run 1: 33.4 / 42.6). [MEASURED]

## GPU

Same probes as the earlier Vulkan lab project; this run agrees with it within ~2%: fp32 wave64 50.0 TFLOP/s, wave32
26.5, packed fp16 53.5, matrix fp16 211 / bf16 205 / fp8 391 / int8 398, VRAM latency ~310 ns,
VRAM bandwidth 640-708 GB/s, best network layout W = 64 in registers 11.8 TFLOP/s. [MEASURED]

## Crossovers

us per run; "with transfer" = CPU writes inputs into VRAM, dispatch, GPU copies the result back,
CPU reads it. GPU results checked against the CPU at every size (bit-equal for map, trees,
chains; within 1e-4 x inputs for nets); mutation-checked. [MEASURED]

| task | GPU (with transfer) beats best CPU from | GPU (resident) from | largest size |
|---|---|---|---|
| map, y = a*x + b | never (64M: 38.6 ms vs 10.9 ms all threads) | 16M floats | resident 0.92 ms, 12x all threads |
| nets, 64 wide x 4 layers x 16 inputs | 64 networks | 64 networks | 4096: 3.6x (transfer), 30x (resident) |
| trees, 128 x depth 8 | 4K rows | 4K rows | 2M rows: 2.5x 16 threads |
| chains, 1,000 steps each | 64K chains | 64K chains | 1M: 7.8x |
| chain-long, one chain | never | never | CPU 1.15 ns/step; GPU 7.7-15.8 ns/step |

- Trees at 256 rows: GPU kernel 313 us: one lane walks 1,024 dependent nodes, each waiting on
  memory (~300 ns) [DERIVED from the latency map].
- map at 16M floats (64 MiB in + 64 MiB out) is where the CPU leaves L3: all threads 47 us at 4M
  -> 1,050 us at 16M. [MEASURED]

## Test-time trap: spinning start barrier under ThreadSanitizer (2026-10-15)

The CPU probes' start barrier spun on an atomic flag. Under ThreadSanitizer every atomic load
takes a lock inside TSan, and 16 pinned spinning workers starved the main thread that sets the
flag: the stream probe's test never finished (> 9 min at 1600% CPU). Spin 4,096 times, then
yield: the test takes 7 s under TSan, and the ReleaseFast alu rows came out identical (106 rows,
median change 0.0%). [MEASURED]

## Repeatability: 4 full runs (2026-10-15)

`docs/data/runs/run1-4.csv` (+ `run5-cpu.csv`), compared by `zig build spread` into
`docs/data/strata-spread.csv` (median, min, max, spread = (max - min) / median per row). Run 2's
CPU rows are left out (a compile ran next to them) and replaced by the CPU-only run 5, so every
row has 4 runs. 723 rows. [MEASURED]

- **Every crossover verdict is the same in all 4 runs** (map: GPU only when resident, from 16M;
  nets 64; trees 4K; chains 64K; chain-long never), and the margins at the largest size agree
  within 2% (nets 3.65-3.68x, trees 2.55-2.59x, chains 7.77-7.90x).
- **Within 2% across runs:** all throughput peaks and streaming numbers above the L1 range: CPU
  ALU at every width and thread count, GPU ALU / lanes / occupancy / shared memory / matrix units /
  networks, large uploads and downloads, CPU latency at every level, RAM 57-58 GB/s, VRAM 640 GB/s.
- **Noisy, read as rough:** anything dominated by waking or scheduling: a parked pool's wake-up
  (12-37 us; 67% spread at 2 threads), one GPU round trip with a dispatch (36.7-42.5 us), the
  small-size gpu-resident times in the crossovers (58-102 us, two clusters near 60 and 100); GPU
  latency where the working set spills out of the Infinity Cache (64-200 MiB: up to 43%) and VRAM
  latency itself (290-327 ns); CPU streaming at 8-32 KiB.
- [DERIVED] The noisy rows are the ones where the desktop (which shares the GPU and CPU) or clock
  ramp-up can matter; none of them changes a placement rule.

## Native Windows 11: 4 full runs (2026-11-08)

`docs/data/runs/run7-10-windows.csv`, spread in `docs/data/strata-spread-windows.csv`. Same
machine, Windows 11 Pro 26200, AMD's Vulkan driver (not RADV: "AMD proprietary driver 26.10.1
(LLPC)", Vulkan 1.4.359; full capability list in `docs/data/device-windows.txt`, from
`zstrata --device-info`), Zig 0.17.1; the process runs at
BelowNormal priority in a job object, on a desktop that was idle but in use. Compared with the
Linux set above (RADV, Zig 0.16: GPU/link/cross from runs 1-4, CPU from runs 1, 3, 4, 5). A
difference counts only when the 4 Windows runs and the 4 Linux runs do not overlap. 698 rows:
the 25 Linux rows that need more shared memory than this driver allows are skipped. [MEASURED]

- **Every crossover verdict is the same as on Linux** (map: GPU only when resident, from 16M;
  nets 64; trees 4K; chains 64K; chain-long never). Margins at the largest size are a little
  wider: nets 4.2x with transfer / 46x resident (Linux 3.6x / 30x), trees 2.7x (2.5x), chains
  10.8x (7.8x), map resident 11x (12x).
- **Throughput peaks: ~2-3% lower, and consistently so.** CPU AVX-512 0.328 vs 0.335 TFLOP/s on
  8 cores; GPU fp32 46.9 vs 48.3 TFLOP/s; same for lanes and occupancy. VRAM streaming
  640 GB/s on both; 256 MiB 799 vs 707 GB/s (old stream probe: its cache rows were wrong, see
  "GPU cache bandwidth, corrected").
- **CPU memory:** RAM latency 92 vs 88 ns (+5%), RAM bandwidth 57 vs 58 GB/s (all threads),
  one thread 71 vs 61 GB/s. Around L3 Windows is faster: 1 MiB 4.1 vs 5.6 ns, 32 MiB 12.6 vs
  15.4 ns. One thread at 4 KiB reads 633 GB/s (619-634) vs 334: Windows reaches the L1 peak of two
  64-byte loads per cycle [DERIVED]; why the Linux runs got half is not known.
- **Fresh memory: 2x faster.** First write to 1 GiB 11.1 vs 4.6 GB/s, writing it again 20.9 vs
  10.4 GB/s.
- **Threads: spawn + join 93 us vs 9 us (10x).** WSL2 measured the same 99 us. About 40 us of
  it is Zig's default stack, committed up front on Windows (next section). Hot pool round trips are equal or faster up to 8 threads
  (0.12-0.48 us) and 1.6 us vs 1.0 at 16.
- **Pool after idle is not comparable in these runs:** the pool (then zarbor's) parked idle
  workers only on Linux (futex); on Windows they kept polling, so its 0.4-0.7 us (Linux 20-29 us)
  was bought with busy CPUs. Fixed in the next section.
- **GPU submission costs ~30% more:** empty submit 44.6 vs 33.8 us, one dispatch with readback
  47.6 vs 36.8 us. Small uploads are slower (4 KiB GPU copy 3.4 vs 6.4 GB/s); staged downloads of
  1 MiB are 2.2x faster.
- **The shader compiler differs, and it shows on the hard kernels.** One long dependent chain on
  the GPU: 2.2-2.9x faster on Windows (1M steps: 7.2 vs 15.9 ms). MLPs with weights in shared
  memory: about half the speed (0.52-0.66x at every width; our shader's load width, fixed in the
  next section); weights from memory swing with the
  width, 0.12x (W=256) to 6.7x (W=64). Each side is within 1% across its own runs, so these are
  the drivers, not noise.
- **Driver limits:** 32 KiB of shared memory per workgroup (RADV: 64 KiB), so occupancy above
  32 KiB and the larger shared-memory MLPs are skipped; buffers up to 2 GiB. Per-shader
  statistics exist under other names (`numUsedVgprs`, `numUsedSgprs`, `ldsSizePerLocalWorkGroup`,
  `scratchMemUsageInBytes`); runs 7-18 left those columns empty because zstrata only looked for
  RADV's names, fixed after run 18. AMD reports no spill, occupancy, VOPD or SMEM counts.
- **Same capabilities otherwise:** every extension the probes use is present
  (cooperative matrix with fp16, bf16, fp8 e4m3/e5m2, int8; bfloat16; float8; subgroup size
  control 32-64; pipeline statistics), and the matrix units run at the same rates (fp16 214 vs
  208, fp8 389 vs 389 TFLOP/s, int8 397 vs 399 TOP/s).
- **Noise on Windows:** CPU latency and GPU rows are as steady as on Linux (median spread under
  3%); wake-ups, crossovers at small sizes and small uploads/downloads move more (up to 2x).
- [DERIVED] Placement rules from this map hold on Windows unchanged. What changes: avoid creating
  threads per task (10x), batch GPU submits a little harder (+11 us each), keep a shared-memory
  workgroup at 32 KiB or read the limit at run time.

## Windows gaps: ours or the OS? (2026-11-08)

Each Windows deficit above was tested for a cause in this code (or Zig's std) before calling it
the OS or the driver. `docs/data/runs/run11-14-windows.csv` (spread
`docs/data/strata-spread-windows-fixed.csv`) are 4 full runs after the fixes, with zstrata's own
pool (`src/cpu/pool.zig`; zstrata no longer depends on zarbor). Every crossover verdict is still
the same. [MEASURED unless tagged]

| gap (runs 7-10) | cause | whose | after |
|---|---|---|---|
| shared-memory MLPs 0.52-0.66x | the shader read shared memory one float at a time; AMD's compiler pairs them (`ds_load_2addr_b32`, seen in RGA 2.15 live-driver ISA) | ours | new `lds4` variant reads vec4 (`ds_load_b128`): 1.01-3.24 TFLOP/s, 1.3-1.8x the Linux scalar numbers; W=8 single net 0.50 vs 0.59 (worse) |
| idle pool workers never slept | the pool parked on Linux only | ours | parks on Windows with RtlWaitOnAddress: after idle 10.6 / 16.2 / 24.5 us at 2 / 4 / 8 threads (Linux 20-29 us) |
| spawn + join 93 us | Zig's default 16 MiB stack is committed up front on Windows (CreateThread with 16 MiB committed: 108 us, reserved: 75 us) | Zig std | `spawn-join-1mib`: 60 us. The rest (CreateThread floor ~71-75 us vs Linux 9) is Windows |
| GPU round trip +30% | recording 1-6 us; waking the thread blocked in vkWaitForFences ~7 us | Windows wake-up | prerecorded + polling the fence: 35.9 us, at the Linux level (33.8, blocking); costs a busy core |
| CPU peaks -2% | ~90 MHz less boost: the same 0.98x at 1 chain (latency-bound) and at 64, every width; power plan already Ultimate Performance [DERIVED] | CPU boost | none from code |
| 16 threads ~0.9x | every logical CPU busy, so any other runnable thread on Windows preempts one of ours (these runs are at BelowNormal priority) [DERIVED]; 12 threads: 0.995x | process priority | none from code |
| memory-weight MLPs 0.12-6.7x | the two compilers pick opposite register strategies (RGA: W=64 165 VGPRs on Windows vs 48 RADV; W=256 33 vs 256 + spills) | driver compilers | none from code; the same SPIR-V |
| small GPU copies, 32 KiB shared limit, no register stats | AMD driver | driver | memory types checked: staging = system RAM write-combined, readback = cached system RAM, direct = mappable VRAM, as intended |

- Raw shared-memory bandwidth (`lds`) is 0.91-1.00x of Linux, so the MLP gap was never the
  hardware path. The Linux `lds4` numbers are not measured yet.
- Short spot checks right after a rebuild read high (spawn 108-140 us): a fresh executable runs
  slower at first [DERIVED: Defender scanning it is the likely cause]. The full runs agree with
  each other (spawn 89-98 us over runs 7-14). Judge from full runs only.

## Speed experiments on Windows (2026-11-08)

`docs/data/runs/run15-18-windows.csv`, spread `docs/data/strata-spread-windows-speed.csv`: 4
full runs with the new rows (744 rows each). What the rows say and how to use them in other
projects is in [`speed-playbook.md`](speed-playbook.md). Headlines [MEASURED, Windows only]:

- **Batching** (`link,batch`): dependent dispatches in one submit cost 12.5 / 4.1 / 1.5 / 0.85 us
  each at 4 / 16 / 64 / 256 per submit, against 41-48 us with a submit + wait each.
- **Waiting** (`link,wait`, a 2.9 ms job): block 41 us of CPU, spin 2883 us (the whole job),
  hybrid (poll 50 us, then block) 91 us; wall time within 25 us across all three. Empty round
  trip, prerecorded: block 49.4, spin 44.2, hybrid 44.1 us. The waiting thread's CPU on Windows
  comes from QueryThreadCycleTime calibrated against QPC (GetThreadTimes counts whole 15.6 ms
  ticks); spin's CPU equals its wall time, which checks the calibration.
- **vec4 loads** (`gpu,mlp`, `lds4`/`mem4`): 1.4-7.2x over the float versions at most widths,
  0.85-0.97x at W=8 and 0.85x for 2 packed nets at W=32 (buffer). Old variants compile to
  byte-identical SPIR-V, so their numbers stand.
- **Best placement per width** (every `mlp` variant): registers win from W=8 to W=192 on both
  OSes; best Windows / best Linux 0.95-1.18x, except W=32 0.84x, W=96 0.69x, W=128 0.78x.
- Round trips were higher in these 4 runs than in runs 7-14 (empty submit 53.6 us, range
  43.7-66.6): read round-trip rows as within-run comparisons, not absolutes.

## Repeat on Windows, and how to tell a disturbed run (2026-11-08 evening)

`docs/data/runs/run23-26-windows.csv` (spread `docs/data/strata-spread-windows-run23-26.csv`),
d8ec1c1, desktop idle. Against runs 15-18 the CPU, GPU and crossover rows agree: median ratio
per probe 0.99-1.03, crossover verdicts unchanged (nets 64, trees 4K, chains 64K, map only
resident, chain-long never). Against Linux the picture of the earlier sections holds: AVX-512
0.98x, RAM latency 92.6 vs 88.2 ns, 1 MiB 4.9 vs 5.6 ns, one thread at 4 KiB 633 vs 335 GB/s,
spawn + join 98 vs 9 us, GPU fp32 0.99x, matrix units 1.02-1.03x. [MEASURED]

- **GPU submission drifted up over the day, and it is not this code.** Empty submit 44-50 us
  (runs 7-16), 63-67 (runs 17-18), 66-146 (runs 23-26, two runs near 70 and two near 145).
  d8ec1c1 adds nothing on the submit path, and the drift started on the commit before it. In the
  slow runs even the 2.9 ms job takes 5% longer (3044 vs 2902 us wall), which points at the GPU
  or driver state (clocks, power) [DERIVED]; not checked further. In runs 25-26 the slow round
  trips moved nets-with-transfer from 64 to 256. Batching is still the lever: 256 dispatches in
  one submit 0.93-1.24 us each against 73-150 us with a submit each.
- **Disturbed runs.** Runs 19-22 (same day, the PC in use while the runs are at BelowNormal priority)
  are not kept. Signs: 4 KiB chase 2.6-3.4 ns (clean 0.96-0.99), all-thread small tasks 2-6x
  slower (`cross,chains,cpu-all,4096` 54-144 us, clean 23-53), and the crossovers moving to
  *smaller* sizes (nets 16, trees 1K, chains 4K) because the CPU side got slow, not the GPU fast.
  Check the 4 KiB and 1 GiB chase rows (0.96-0.99 ns, 91-95 ns) before using a run.

## Where a round trip goes, and what idle does to it (2026-11-08 evening)

New probes `link,phases` and `link,idle` (GPU timestamps placed on the CPU clock with
VK_KHR_calibrated_timestamps; the driver bounds its own error at 17-45 ns). Data:
`docs/data/runs/phases-idle-{1,2,3}-windows.csv` (`link --only phases,idle`, plus one earlier
spot run not kept) and the full run `run27-windows.csv`. 4-5 runs, Windows only. [MEASURED
unless tagged]

| one tiny dispatch, blocking wait | median of 200 trips (range over runs) |
|---|---|
| record the command buffer | 1.9-2.0 us |
| inside vkQueueSubmit2 | 12.4-12.5 us |
| launch: submit called -> GPU starts | 41-46 us (one run: 45) |
| run on the GPU | 0.40-0.44 us |
| wake: GPU done -> CPU thread running | 27-28 us (polling: 22-25 us) |
| total | 70-75 us (one run: 96) |

- **The GPU's own work is under 1 us of a ~70 us trip.** Even polling the fence, "GPU done ->
  fence seen" takes 22-25 us, so most of the wake-up is the driver signalling the fence, not
  the sleeping thread [DERIVED]. Launch is the largest part.
- **Launch is where the drift sits.** It is bimodal between runs: 41-46 us, or ~2x that
  (103-150 us in one spot run in four, cold or warm). GPU run time does not change with it. The
  70/145 us split in runs 23-26 matches. Cause: driver or GPU front-end state [DERIVED]; not
  known.
- **After the thread sleeps 10 ms or more, the CPU side slows 5-15x:** recording 11-31 us
  (2 us), the submit call 46-68 us (12.5 us), total after 1 s idle 166-240 us. One throwaway
  trip after the idle ("warm") recovers part of it (1 s: 118-167 us). Core clock or idle states
  on the CPU [DERIVED: the GPU-side run time is unchanged at 0.68-0.76 us].
- **Disturbance check, first full run (27):** 5 checkpoints, worst move 4.5% (FMA per core),
  flagged clean. L1 0.985-0.992 ns, RAM 94.9-97.3 ns, FMA 2.44-2.55 TFLOP/s.
- `zig build test` broke on this machine between 12:40 and 17:40 without a code change: this
  account lost read access to `C:\Program Files (x86)\Steam`, and the Vulkan loader logs an
  error for Steam's registered layers. Loader messages went to progress output (quiet in
  tests) instead of stderr; they were never counted as validation messages. Since 0.2.0
  (2026-11-25) they are dropped unless `gpu.show_loader_notes` is set, and progress output
  is off in every test build, so projects using the Vulkan layer stay quiet too.

## What a portable build costs (2026-11-08 evening)

zstrata built five ways (`-Dcpu=native` (znver5), `x86_64_v4`, `x86_64_v3`, `x86_64_v2`,
`baseline`), `cpu cross` run 3 times each, interleaved (all five, then again, three rounds).
Data `docs/data/runs/isa/`, one spread file per build. GPU kernels are SPIR-V and do not depend
on the CPU target; only the CPU rows and the CPU side of the crossovers can change. [MEASURED
unless tagged]

| row (median of 3) | native | v4 | v3 | v2 | baseline |
|---|---|---|---|---|---|
| FMA, 1 thread, scalar (TFLOP/s) | 0.0197 | 0.0196 | 0.0197 | 0.0020 | 0.0025 |
| FMA, 1 thread, 8 lanes x 16 chains | 0.163 | 0.164 | 0.100 | 0.0019 | 0.0025 |
| FMA, 1 thread, 16 lanes x 16 chains | 0.326 | 0.326 | 0.114 | 0.0018 | 0.0024 |
| FMA, 8 threads, 16 lanes | 2.57 | 2.53 | 0.86 | 0.014 | 0.017 |
| read 4 KiB, 1 thread (GB/s) | 615 | 637 | 322 | 62 | 62 |
| read 1 MiB, 1 thread | 222 | 220 | 186 | 61 | 63 |
| read 1 GiB, 1 thread / all threads | 55 / 55 | 57 / 56 | 57 / 59 | 55 / 59 | 55 / 59 |
| RAM latency (ns) | 94.2 | 94.1 | 92.8 | 94.1 | 93.8 |
| nets on the CPU, 4096 nets, 1 thread (ms) | 129 | 130 | 185 | 1558 | 1442 |
| nets, all threads (ms) | 14.6 | 15.3 | 17.7 | 170 | 158 |
| map 64M, 1 thread / all threads (ms) | 12.0 / 11.5 | 12.1 / 11.4 | 12.1 / 11.5 | 68.1 / 12.0 | 56.7 / 11.8 |
| trees 2M rows, all threads (ms) | 367 | 373 | 368 | 370 | 364 |
| one long chain, 10M steps (ms) | 12.0 | 11.9 | 12.7 | 14.4 | 14.4 |

- **v4 is native** for every probe (median ratio per probe 0.98-1.07). Zen 5 scheduling and the
  extra features above v4 (VNNI, BF16, ...) buy nothing here, because no probe uses them yet.
- **v3 (AVX2, the usual portable target) costs on vector math, not on memory.** 16-lane code
  runs at 0.35x (one thread) and 0.33x (8 threads); 8-lane code with 16 chains at 0.61x, so
  register pressure (16 vector registers instead of 32) costs on top of the width [DERIVED].
  L1 reads 0.52x (32-byte loads), RAM-bound rows equal. Nets on the CPU 1.44x / 1.21x slower.
- **v2 and baseline have no FMA instruction:** `@mulAdd` (exact fused rounding) becomes a
  software routine [DERIVED], so every FMA row falls 10-190x, nets 11-12x, a single-thread map 5x;
  with all threads the map is RAM-bound again. Reads are 0.1x at L1 (16-byte SSE loads).
- **Crossovers:** native, v4 and v3 give the same verdicts (nets 64, trees 4K, chains 64K, map
  only resident from 16M, chain-long never). On v2/baseline the GPU wins nets from 1 and the
  resident map from 1M, because the CPU side got slow.
- Two runs (v3 and v2, round 2, 18:33-18:35) flagged themselves as disturbed on the FMA marker
  alone (14-16% move; L1 and RAM within 6%). In those builds the marker runs slower code paths,
  so this may be the build's own noise; the medians of 3 are unaffected by one such run.

## GPU cache bandwidth, corrected (2026-11-09)

The old `gpu,stream` rows (variant `read-vec4`, every run up to run 27) overstated the GPU's
cache bandwidth 2-5x. Two causes, found by testing each in a scratch copy:

- **The grid did not fit on the GPU.** 1024 workgroups of 256 lanes; the card holds about 512
  at once [DERIVED: 128 SIMDs x 16 wave slots = 2048 wave64 slots, 4 per workgroup], so the grid
  ran in two batches and each batch read only part of the working set. Measured with a strict
  sweep: at 1024 workgroups every cache step sat at twice its real size (L2 knee at 16 MiB,
  Infinity Cache at 128 MiB); at 256 and 512 they sit at 8 and 64 MiB, the published sizes. The
  old order (jump = workgroups + 1) also let workgroups re-read blocks others had just read,
  which inflated the in-cache rows further (10-12 TB/s at 8-16 MiB vs 6-8.5 strict).
- **The buffer was all zeros.** RDNA4 compresses memory; zeros read 9-23% faster than random
  data from the caches, the same from VRAM (3 interleaved runs each, idle desktop; a first set
  taken while a game ran showed up to 1.55x and was discarded).

The probe now fills the buffer with seeded random bytes, sweeps blocks in one strict order
(`shaders/stream.comp`, checked by the test "gpu stream reads the blocks in strict sweep order",
3 of 3 mutants killed), takes any size in whole 4 KiB blocks, and runs grids of 256 and 512
workgroups (`y` column). New variant name `sweep`, so old and new rows never mix.
Data: `docs/data/runs/stream-sweep-{1..4}-windows.csv`, spread
`docs/data/strata-spread-windows-stream-sweep.csv`; CPU markers before / after: 0.99 / 1.00 ns,
99 / 96 ns. [MEASURED, Windows]

| level | holds | 256 workgroups | 512 workgroups | spread |
|---|---|---|---|---|
| nearest caches | up to 2 MiB in all (32 KiB per CU x 64 [DERIVED]) | 21.1-22.4 TB/s | 21.6-21.7 TB/s | <= 3% |
| L2 | 3-8 MiB | 8.5 TB/s | 6.1-6.2 TB/s (15.2 at 3 MiB) | <= 1% |
| Infinity Cache | 12-64 MiB | 2.29-2.34 TB/s | 2.26-2.34 TB/s | <= 3% |
| VRAM | 96 MiB up | 636-639 GB/s | 635-640 GB/s | <= 1% |

- The Infinity Cache reads 2.3 TB/s, 3.6x VRAM, not the 5-13 TB/s the old rows showed. Third
  parties estimate about 2.5 TB/s for it (Chips and Cheese, via a research summary; the page
  itself was not read, so [unverified]).
- Against the CPU (all threads, Windows runs 23-26): GPU L2 6-8.5 vs CPU L2 1.6-2.3 TB/s,
  Infinity Cache 2.3 vs CPU L3 0.65-1.0 TB/s: the GPU's caches lead by 2-5x, not 7-10x.
- The L2 reads faster with fewer workgroups in flight (8.5 vs 6.1 TB/s); not explained.
- Rules that used the old numbers: `experiments.md` B4 (corrected), the page's GPU levels
  (corrected; its Linux GPU bandwidth curve came from the old probe and is left out). The
  Linux `gpu,stream` rows in `strata.csv` stay as recorded but should not be used.

## Clock tracker, and Windows power throttling (2026-11-09)

`cpu,clock` (new, C4): a chain of dependent one-cycle register adds gives the core clock.
5.02 GHz with one core busy, 4.89 with all 8; at the five checkpoints of full run 28 (clean,
worst marker move 4.4%) it stayed at 4.96-5.06 GHz, so that run's RAM-latency move (95.7-99.1
ns, 3.6%) was not the clock. [MEASURED, `docs/data/runs/run28-windows.csv`]

`--no-throttle` (new) opts the process out of Windows power throttling (execution speed and
ignored timer requests, SetProcessInformation, no privilege needed). Hypothesis from the
research: the runs are at BelowNormal priority, which Windows may treat as low quality of service,
and that could explain slow wake-ups and the CPU side slowing 5-15x after idle. Tested with
`link --only phases,idle` and `cpu --only sync`, 3 runs each way, interleaved
(`docs/data/runs/no-throttle/`). [MEASURED]

- **No effect on this desktop.** Every row's 3 values overlap between the two settings: launch
  41.0-41.7 us, fence wake 25-29 us, total 68-73 us; pool after idle 10-13 / 16-20 / 24-26 us
  at 2 / 4 / 8 threads; the first trip after 1-100 ms idle 248-291 us either way.
- So the after-idle slowdown is not Windows throttling the process. Its largest part is the
  GPU launch after idle (cold 204-224 us vs 41 warm-up-free; one throwaway trip brings it to
  55-66 us); recording and submit grow too (2 -> 10 us, 12 -> 41-47 us) [DERIVED: CPU and GPU
  idle states; the clock tracker cannot see sub-millisecond C-states].
- The bimodal launch of runs 23-26 (41 vs ~2x) did not appear today: every launch row was
  41-42 us. Still unexplained.
- Kept the switch: on a laptop or on battery, where Windows does throttle, it may matter.

## Linux gaps: ours, the driver's or the OS's? (2026-11-26)

Native Fedora 45 (kernel 7.3.4, RADV, governor and EPP `performance`), same machine. Before:
`docs/data/runs/run29-32-linux.csv` (4 full runs at c02908a, screen on). After:
`run33-36-linux-warm.csv` (4 full runs with the fixes and `--gpu-warm`; no run flagged
disturbed, worst marker move 1.4-4.9%). Windows: runs 23-26. Compare with
`python3 tools/linux/cmp.py WINDOWS... -- LINUX...`. Medians of 4. [MEASURED unless tagged]

| gap before (Linux vs Windows) | cause, and how it was shown | whose | fix | after (Windows) |
|---|---|---|---|---|
| CPU read bandwidth, working set in L1: 334 vs 633 GB/s; all threads 2.1 vs 4.7 TB/s | the 1 GiB buffer came from glibc `malloc` (Zig's allocator when libc is linked), which returns it 16 bytes into a cache line, so every 64-byte load touched two lines. A stand-alone test at offsets 0/16/32/48: 648 / 334 / 334 / 334 GB/s | ours | `lib/mem.zig` `allocBig`: page-aligned | 643 GB/s; 5.2 TB/s (4.7) |
| first write to fresh memory 8.5 vs 10.9 GB/s; latency 32-64 MiB 15-18 ns, 1 GiB 89 ns | 4 KiB pages: transparent huge pages are `madvise`-only here and nothing asked. The same test with `MADV_HUGEPAGE`: 8.4 -> 22-24 GB/s | OS default, ours to ask | `allocBig` asks for 2 MiB pages | first write 24.1 (10.9); latency 64 MiB 10.8 ns (24.1), 1 GiB 75 ns (93). `first-write-small-pages` keeps the 4 KiB number: 8.2 |
| short GPU jobs ~2x slower: one chain of 1M steps 16.7 vs 7.2 ms, GPU ALU at 1-4 values per lane 0.45-0.55x, a 2.9 ms job 5.3 ms | the GPU clock. With power level `auto` the clock (hwmon `freq1_input`, sampled per row by `tools/linux/clkrun.py`) sits at ~1.6 GHz when work starts, climbs only after ~0.5 s of unbroken load to 3.2-3.4 GHz, and is back down ~0.1 s after the load lightens. Time per step tracks it exactly: 16.7 ns at 1615 MHz, 11.4 at 2370, 8.3 at 3230 (27 cycles each) | amdgpu power management | `--gpu-warm` (`gpu/warm.zig`): one lane kept busy on the second compute queue. No root needed | chains kernel 5.8 us (9.0); nets kernel 187 us (235); 2.9 ms job 1.6 ms (3.0) |
| long chain still 1.16x slower at full clock | RADV's loop: `s_wait_alu`, compare and branch every step (27 cycles per step; AMD's compiler ~24 [DERIVED from 7.2 ns]) | RADV | `chains.comp` does 4 steps per loop round | 5.5 ns/step (7.2; Windows measured with the old shader) |
| GPU scalar load latency 20.4 vs 11.7 ns | RADV loads the buffer's descriptor again at every use (`s_load_b128` inside the loop, `RADV_DEBUG=shaders`), and the scalar data load waits for it; AMD's compiler hoists it | RADV | `chase.comp` does 32 steps per loop round (RADV reuses the descriptor within a round) | 11.72 ns (11.96) at 4 KiB; 13.2 (11.7) at 16 KiB |
| GPU ALU at <= 8 values per lane 0.85x after the clock fix | the same loop header | RADV | `fma.glsl` does 4 rounds per loop pass | K=4 41.9 (39.3), K=8 45.5 (47.0) TFLOP/s; wave32 K=8 23.7 (25.8) |

Not fixed:

- **Network kernels with weights read from memory (`mem`, `mem4`): 0.1-0.5x at W=32-128.** AMD's
  compiler moves the loop-invariant weight loads into registers (165 VGPRs at W=64), RADV
  keeps loading (48). Written that way by hand it is the `reg` variant, and best placement
  against best placement Linux is 0.95-0.98x at W=8-48, 1.0-1.5x at W=32, 64, 96, 128 and
  0.87x at W=192-256. `lds4` is 0.6-0.85x of Windows, `lds` 1.7-1.9x.
- **Pool round trip, workers awake: 0.149 vs 0.123 us (2 threads), 0.54 vs 0.49 (8).** Bimodal on
  Linux: batches of 20,000 round trips read ~120 or ~150-160 ns and switch between the two
  within one process, also with both threads pinned (pairs 0,1 / 2,5 / 3,4; the SMT pair 0,8
  stays at ~120). Both cores hold 5.22-5.23 GHz throughout (add-chain clock on each), so it
  is not the clock or the placement. Polling without `pause` for the first 256 spins did not
  settle it (reverted). Cause not found.
- **Sustained GPU peaks 2-4% lower** (matrix units 206 vs 215 TFLOP/s). Under full load the
  clock reads 3.1-3.2 GHz; Windows' clock under the same load is not known. Not investigated.

What the warmer costs, on the same code (runs 37-40: 4 full runs without it, same afternoon,
screen off like runs 33-36). Windows advantage, median over a probe's sizes, >1 = Windows
ahead:

| rows | without | with `--gpu-warm` |
|---|---|---|
| `chain-long` kernel / `chains` kernel | 2.05 / 1.45 | 0.78 / 0.65 |
| `nets` kernel / `trees` kernel | 1.23 / 1.42 | 0.55 / 0.90 |
| 2.9 ms job, wall (`wait`) | 1.32-1.35 | 0.54-0.55 |
| empty round trips | 0.28-0.43 | 0.13-0.19 |
| staged download | 1.15 | 0.63 |
| staged upload | 0.81 | 0.94 |
| small jobs with the round trip (`chains` / `nets` / `map` resident) | 0.40 / 0.53 / 0.50 | 0.68 / 0.75 / 0.66 |
| `map` with transfer | 0.84 | 1.05 |

So the unrolled shaders alone do not close the kernel gaps: without the clock they stay
1.2-2x behind. The warmer loses on staged uploads from 256 MiB (9.7 vs 13.1 GB/s; the GPU-side
copy and direct writes are unchanged), on the launch phase of a round trip (20 -> 71 us) and
on tiny jobs timed with their round trip (still ahead of Windows). It stays a flag. Learned
while building it:

- 8 ms warmer jobs slowed other work's GPU cache reads at 1-8 MiB by up to 1.5x: every job
  boundary flushes the GPU's caches. ~0.2 s jobs: no difference.
- A 4 s warmer job stalled the other queue's jobs for good (both fences never signalled).
- A job cannot be stopped through a flag in a buffer: the GPU keeps its cached copy.
- The amdgpu "stable pstate" call (`tools/linux/pstate.py`; any user may make it) pins the
  clock at 2370 MHz and the memory clock at its top: better than a cold start, worse than the
  warmed 3.2-3.4 GHz.

**Screen off changes GPU memory numbers.** From ~15:15 the desktop's screen was locked and the
display off; runs 33-36 are from that time. GPU latency past the caches (96-512 MiB) read
430-850 ns instead of 200-330: the memory clock (hwmon `freq2_input`) sat at 96-456 MHz
instead of 1258. Holding the stable pstate (memory clock 1258 MHz) brought 512 MiB back to
264 ns. So in runs 33-40 the `gpu,chase` rows from 96 MiB are not comparable; measure GPU
memory with the screen on.

**The memory clock also explains the morning's gap.** GPU latency past the caches, ns
(`docs/data/runs/gpu-mem-peak-pstate-*-linux.csv`, 2 runs under the stable pstate):

| ring | Windows | Linux, screen on (runs 29-32) | Linux, screen off | Linux, stable pstate |
|---|---|---|---|---|
| 96 MiB scalar / vector | 191 / 183 | 227 / 225 | 239 / 422 | 206 / 209 |
| 128 MiB | 215 / 202 | 267 / 296 | 245 / 619 | 211 / 202 |
| 256 MiB | 238 / 235 | 318 / 321 | 270 / 747 | 252 / 247 |
| 512 MiB | 248 / 240 | 363 / 350 | 294 / 664 | 255 / 265 |
| 1 MiB (in cache) | 61 / 77 | 61 / 77 | 61 / 76 | 85 / 112 |

With the memory clock held at its top Linux is within 0-10% of Windows past the caches; with
the default power level it was 1.2-1.45x behind even with the screen on, because one lane of
reads is not enough load to raise that clock [DERIVED: the clock was only sampled with the
screen off]. The stable pstate pays for it in the caches (core clock fixed at 2370 MHz). Both
clocks at their top at once needs a power profile set by root; not tested.

The shader changes also change what Windows would measure: `alu`, `lanes`, `occupancy`,
`gpu,chase` and the GPU rows of `chains` / `chain-long` need a Windows rerun before the
comparison is like for like.

## Linux: GPU power settings set from the sandbox (2026-11-27, first pass)

The sandbox account can now write `power_dpm_force_performance_level` and
`pp_power_profile_mode` of the RX 9070 XT (host udev rule). The same short rows under each
setting, no warmer, screen on, 20 s idle before each, one run each
(`tools/linux/profiles.sh`, `proftab.py`; `docs/data/runs/gpu-power/`). Kernel 7.3.6.
[MEASURED, single runs: a first pass, not medians of 4]

| row | Windows | auto, default profile | level `high` | custom profile, GFXCLK min active 3300 |
|---|---|---|---|---|
| chain of 1,000 steps, kernel (us) | 8.96 | 12.0 | 8.24 | 5.92 |
| chain of 1M steps (us) | 7156 | 11590 | 7925 | 5502 |
| chain of 10M steps (us) | 69940 | 58130 | 79010 | 54860 |
| 1M chains, kernel (us) | 202 | 293 | 198 | 161 |
| 4096 nets, kernel (us) | 235 | 363 | 249 | 216 |
| scalar load, 4 KiB (ns) | 11.96 | 24.8 | 16.9 | 11.8 |
| shared-memory bandwidth, stride 1 (GB/s) | 12120 | 12580 | 9543 | 12620 |
| staged upload, 1 GiB (GB/s) | 13.9 | 14.0 | 13.9 | 13.8 |
| empty round trip (us) | 110 | 16.0 | 18.7 | 16.1 |
| GPU latency, 512 MiB, scalar / vector (ns) | 248 / 240 | 348 / 329 | 262 / 260 | 427 / 393 |
| idle after 20 s | | 1.55 GHz, 27 W | 2.5 GHz, 76 W | 2.5 GHz, 66 W |

- **Custom profile** (`echo "6 0 0 4 3300 4 800 4587520 -65536 0" > pp_power_profile_mode`,
  level `auto`): the core clock is at its boosted level from the first job. Every clock-bound
  row beats Windows without the warmer, and none of the warmer's costs appear (uploads, round
  trips unchanged). It does not raise the memory clock: past the caches it is the slowest
  setting.
- **Level `high`** is not "everything at the top": the memory clock is pinned at 1258 MHz
  (latency past the caches within 5-8% of Windows), but the core clock is fixed near 2.5 GHz
  with no boost, so long and heavy work loses 25-36%.
- **Stock profiles** COMPUTE (5) and 3D_FULL_SCREEN (1) changed nothing measurable here.
- **Fabric clock**: adding `6 1 0 4 2400 1 0 5898240 -6553 -6553` (FCLK min active 2400) to
  the custom profile looked like a gain in one run (512 MiB: 427 -> 311 ns) and was not one
  on a repeat (311 ns with and without it). Run-to-run noise; no effect shown.

So it can be done from this account: the custom core-clock profile replaces the warmer. Not
done yet: 4 full runs under it; the memory clock together with the boosted core clock (the
fabric-clock line gets part of the way); the same with the screen locked; idle power is ~40 W
higher while the profile is set, so it belongs around a run, not left on.

**4 full runs under the custom profile** (`run41-44-linux-custom.csv`, `tools/linux/fullruns.sh`,
no warmer; no run flagged disturbed). Against Windows runs 23-26, Linux is at or ahead on every
probe except, as Windows advantage (median over a probe's sizes): network kernels with weights
in memory or `lds4` 1.2-1.6x; pool round trip 1.16x; staged and direct downloads 1.14-1.15x;
GPU latency past the caches. The screen locked during run 42 and was off for runs 43-44:
512 MiB latency read 287-310 ns in runs 41-42 (Windows 248) and ~750 ns in 43-44, so the set
is mixed for those rows and the memory clock stays the open GPU item. Not tried: pinning it
through `pp_dpm_mclk` with level `manual` (root-only file).

## Ceiling experiments (2026-11-27 evening)

One bounded experiment per open gap (PLAN.md, "Ceiling target"). Governor and EPP `performance`,
idle machine (load 0.4-0.6), card back on `auto` / profile 0 after every step.

- **E6, network kernels with weights in memory.** The per-row pipeline statistics already
  answer it. `mem-one` at W=64: AMD's compiler 165 VGPRs (weights hoisted into registers),
  RADV 48 (reloaded on every use). Copying the rows a lane needs into locals by hand is the
  `reg` variant. Best variant against best variant per network size, runs 23-26 against
  41-44: Linux 0.89-0.98x at W=8-48 and 192, 1.03-1.50x at W=32 packed, 64, 96, 128, 256.
  [MEASURED] Small-W `reg` stays 4-10% behind with equal VGPR counts and `gpu/alu` at parity:
  instruction scheduling in RADV [INFERRED]. W=256 spills 1046 VGPRs on RADV (249 KB
  scratch), but Linux's best there (`mem-one`, 1.11 TFLOP/s) matches Windows' best (`reg-one`,
  1.08). `lds`/`lds4` are never the best variant on either system.
- **E4, pool round trip.** `tools/linux/poolperf.zig` reads hardware counters per batch of
  20,000 round trips, on both threads. Fast (~90-125 ns) and slow (~140-160 ns) batches run
  the same ~150 instructions per round trip. In the slow mode the submitting thread adds about
  1.1 L1 misses served from its own L2 (0.05-0.1 -> 1.2), 1.1 more lines from another core's
  cache (8.3 -> 9.4) and 0.4 more pipeline resyncs (1.6 -> 2.0) per round trip; the helper
  moves the same way, less. [MEASURED] So a hot line is evicted from L1 once per round trip.
  What evicts it was not found; an SMT sibling sharing the L1 is untested. [INFERRED]
- **Downloads.** New variants `download,gpu-copy-only` (timestamps) and `copy-out-only` split
  `staged`. Run alone, 3 runs: GPU copy 21 GB/s from 256 KiB up, copy out 21-57 GB/s, staged
  7.1 GB/s at 1 MiB (Windows 3.7, runs 41-44 2.0) and 10.6 at 1 GiB (Windows 11.8). Linux's
  staged gap shows only inside a full run, after the earlier probes. [MEASURED] `direct-read`
  is 0.073-0.080 GB/s on the default GPU profile (Windows 0.06-0.07) and 0.056 under the
  custom profile: the profile that fixes short GPU jobs costs 25% on CPU reads of VRAM.
  [MEASURED]
- **Staged upload at 64 MiB** (1.14-1.19x behind in both 4-run sets) reads 14.2 GB/s run alone,
  3 runs (Windows 13.65). Like the download, the transfer gap is a full-run effect. [MEASURED]
- **CPU chase at the L2 edge: fixed.** Rings under 2 MiB were on 4 KiB pages (`allocBig` uses
  huge pages from 2 MiB up). The L2 is physically indexed, so a ring scattered over 4 KiB pages
  overfills some L2 sets before it reaches the L2's size. Cut from a huge page instead, 3 runs
  each: 768 KiB 3.73-3.90 -> 2.88-2.90 ns, 1 MiB 5.14-6.00 -> 3.11-3.12, 1.5 MiB 8.74-9.87 ->
  7.77-8.18 (Windows, 4 KiB pages: 3.47-3.67, 4.18-5.00, 8.17-8.72). [MEASURED] The probe now
  does that for every size, so Linux `cpu,chase` rows under 2 MiB from before this change are
  not comparable. For code: a working set near the L2 size reads faster on a huge page, not
  only one past the TLB's reach (playbook rule 15). [DERIVED]
- **Rows dropped as noise** (the ceiling rule: a single row must recur in the next 4-run set):
  `cross/map/cpu-1` (runs 41-44 only), `cross/trees/gpu-with-transfer` (evening set only),
  `cross/map/gpu-kernel` (Linux 1.3x ahead in 41-44, 1.2x behind in the evening set). [MEASURED]
- **Final 4-run set, runs 45-48** (custom GPU profile, current code): every probe/variant median
  is within 1.05x of Windows or accepted; median Windows advantage 0.982. [MEASURED]
- **Since the evening reboot**, three rows that were ahead in runs 41-44 trail, alone as well as
  in full runs. Tiny GPU kernels (`cross/map/gpu-kernel`, 1-64 KiB) are at parity on the default
  GPU profile and 1.10-1.17x behind under the custom one, which put them 1.3x ahead that
  morning. `cross/map/cpu-all` at 64-256 KiB went from 0.91-0.95 to 1.2-1.4, and
  `cross/trees/gpu-with-transfer` at 256 KiB from 1.04 to 1.2-1.4, on either profile. [MEASURED]
  EPP changed from `balance_performance` to `performance` across the reboot; that it causes
  this is a hypothesis only, untested (it needs root). [INFERRED]
