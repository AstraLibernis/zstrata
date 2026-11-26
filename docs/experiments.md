# Experiments not yet run

Ideas to push speed and accuracy further on this machine: Ryzen 7 9800X3D (Zen 5, `znver5`,
one CCD, 96 MiB L3), RX 9070 XT (RDNA4 gfx1201, 16 GiB, 64 MiB Infinity Cache), Windows 11, AMD
driver 26.10.1. Hardware-specific on purpose (owner, 2026-11-08). Nothing here is measured
yet; each item says what it would measure, why it might pay, and who it would serve. Checked
against the code on 2026-11-08: none of these probes exist in zstrata.

"Accuracy" means two things: how much a zstrata number can be trusted (measurement), and how
close a fast result is to the exact one (numerics).

Tags: **[gated]** needs administrator rights or a system setting. **[blocked]**
cannot be done with this setup.

## A. CPU speed

| # | experiment | why it might pay | serves |
|---|---|---|---|
| A1 | **DONE 2026-11-08** (playbook rule 13, `measurements.md`). **Generic vs native builds**: the same probes built `-Dcpu=baseline`, `x86_64_v2`, `x86_64_v3`, `x86_64_v4`, `native` (znver5) | what portability costs; a native build crashes on CPUs without AVX-512 | anything shipped (zmural), every project |
| A2 | **AVX-512 VNNI int8** (`vpdpbusd`) and **AVX-512 BF16** (`vdpbf16ps`) dot products: throughput per core and on 8 cores | the CPU lists both (`avx512vnni`, `avx512bf16`); int8/bf16 is 2-4x the work per instruction of fp32 [DERIVED from the instruction widths] | network inference on the CPU, zarbor |
| A3 | **Is Zen 5 a full 512-bit core here?** AVX-512 vs AVX2 at the same chain count, per instruction class (FMA, shuffle, gather) | the existing `alu` rows cover FMA only; gathers and shuffles decide tree and histogram code | zarbor, CPU math projects |
| A4 | **Large pages (2 MiB)** for the 1 GiB chase and streams | part of the 92 ns RAM latency is TLB misses; large pages remove most of them **[gated: needs the "Lock pages in memory" right, which only an admin can grant]** | all big-memory code |
| A5 | **Non-temporal stores** for big writes, and touching memory up front | first write runs at 11 GB/s, rewriting at 21 GB/s; streaming stores skip the read-for-ownership | zarbor (feature matrices), zscour |
| A6 | **Working set vs the 96 MiB L3**: latency and bandwidth steps at 32 / 64 / 96 / 128 MiB, single and multi-thread | this chip's V-Cache is its main edge; tiles sized to fit it should beat tiles sized for "a normal 32 MiB L3" | zarbor histograms, CPU math projects |
| A7 | **Software prefetch distance** in gather patterns (random index into an array) | tree traversal and histograms are gathers; the hardware prefetcher cannot predict them | zarbor |
| A8 | **Branchless vs branchy** tree walks at real depths and split balances | a mispredict costs ~4.4 ns, as much as ~4 L2 hits | zarbor |
| A9 | **SMT for memory-bound work**: one thread per core vs two, for chase and gather | SMT adds 0 to FMA throughput (measured), but may hide memory latency | any pool sizing decision |

## B. GPU and link speed

| # | experiment | why it might pay | serves |
|---|---|---|---|
| B1 | **MLP layers on the matrix units** (cooperative matrix fp16/bf16/fp8) vs the fp32 register kernels | matrix units do 214 (fp16) / 402 (fp8) TFLOP/s against 25.9 fp32 on the vector ALUs, 8-15x headroom; the MLP probes use none of it | GPU networks |
| B2 | **Overlap transfer and compute**: a second (transfer) queue and double buffering, measured as total time for "upload, compute, download" chains | zstrata uses one compute queue; with overlap, "with transfer" crossovers could approach the "resident" ones (map: 0.3x → up to 11x) | GPU networks, zarbor GPU placement |
| B3 | **Persistent kernel**: one long-running dispatch pulling work from a queue in memory, vs many submits | the submit cost (44-146 µs today) disappears entirely | network training loops, zmural |
| B4 | **Tile to the Infinity Cache**: same kernel with working sets of 16 / 32 / 64 / 128 MiB | GPU reads 21 TB/s up to 2 MiB in all, 6-8.5 TB/s in L2 (8 MiB), 2.3 TB/s in the Infinity Cache (64 MiB), 0.64 TB/s past it (`gpu,stream` `sweep`, corrected 2026-11-09; the numbers here before were 2-5x too high); kernels do not exploit it yet | GPU networks (batch and layer sizes) |
| B5 | **Wave32 dual-issue (VOPD)**: kernels written so pairs of independent instructions can issue together | RDNA4 can issue two vector ops per cycle in wave32; the Windows driver reports no VOPD counts, so RGA ISA would be the check | all compute shaders |
| B6 | **Packed fp16 (fp16x2) MLPs** vs fp32 | 2x the vector-ALU rate for half precision; middle ground between fp32 and the matrix units | GPU networks |
| B7 | **Link ceiling**: the largest upload/download with a transfer queue and big host-visible buffers vs PCIe 5.0 x16's ~63 GB/s theoretical | direct VRAM writes reach 22 GB/s; it is unknown whether that is the link, the copy method or ReBAR | everything that streams data to the GPU |
| B8 | **2:4 structured sparsity** on the matrix units | RDNA4 hardware has sparse matrix instructions; whether this Vulkan driver exposes them is **not known**: check the extension list first | GPU networks |

## C. Measurement accuracy (can a zstrata number be trusted)

| # | experiment | why it might pay |
|---|---|---|
| C1 | ~~**Built-in disturbance check**~~ **DONE 2026-11-08**: `cpu,check` rows (L1, RAM, FMA per core) at 5 checkpoints, `disturbed` flag; `zig build spread` compares runs | runs 19-22 were contaminated and only spotted by hand |
| C2 | ~~**Split the round trip with GPU timestamps**~~ **DONE 2026-11-08**: `link,phases` (record, submit, launch, run, wake) | submit latency drifted 44 → 146 µs today with no code change; this shows which part grew |
| C3 | ~~**GPU idle-state test**~~ **DONE 2026-11-08**: `link,idle`, cold and warm after 0-1000 ms | answer in `measurements.md`: the drift is in the launch, and idle slows the CPU side too |
| C4 | ~~**Clock tracker**~~ **DONE 2026-11-09**: `cpu,clock` (one core busy, all cores busy) and a `cpu,check,clock` row at every checkpoint, from a chain of dependent one-cycle adds | separates "the code got slower" from "the CPU boosted lower" (the -2% Windows CPU gap) without needing admin counters |
| C5 | **Randomised probe order and interleaving**: probes in a shuffled order each run, several short repeats instead of one long one | removes drift within a run (heat, boost, background tasks) from the comparisons |
| C6 | **Defender's first-run cost**: time a fresh executable vs the same file run again | spot checks right after a rebuild read high; this would confirm the cause. An antivirus exclusion for the build folder would remove it **[gated]** |
| C7 | ~~**Linux numbers for the new rows**~~ **DONE 2026-11-26** (runs 29-36, native Fedora) (`lds4`, `mem4`, `batch`, `wait`, `spawn-join-1mib`) | the Windows vs Linux comparison is incomplete without them **[blocked: no Linux with GPU access since WSL was removed]** |

## D. Numerical accuracy (how close a fast result is to the exact one)

| # | experiment | why it might pay | serves |
|---|---|---|---|
| D1 | **Precision ladder on the MLP probes**: fp64 CPU reference vs fp32, fp16x2, bf16, fp16 / bf16 / fp8 matrix units; error and speed in one table | B1's speed is worthless without its error; picks the fastest format that is still accurate enough | GPU networks |
| D2 | **Accumulator precision for sums**: fp32 vs fp64 vs compensated (Kahan / pairwise) sums on AVX-512, speed and error at 1K-100M terms | gradient and histogram sums are where error builds up | zarbor, GPU networks |
| D3 | **Deterministic reduction cost**: fixed-order parallel reduction vs worker-order, CPU and GPU | the memory rules demand fixed order; this measures what it costs | zarbor, GPU networks |
| D4 | **CPU vs GPU bit-equality**: same fma chain on both, count the results that differ | the vec4 kernels were shown bit-equal; tells whether that holds in general or only for that kernel | GPU networks (CPU checks of GPU results) |
| D5 | **Fast-math cost and error**: `@setFloatMode(.optimized)` vs strict on the ALU and MLP probes | measures how much speed strict IEEE costs, and how much error the fast mode brings | all numeric code |

## E. Linux (added 2026-11-26, after the native Fedora runs)

| # | experiment | why it might pay |
|---|---|---|
| E1 | **GPU power profile set by root**: `pp_power_profile_mode` (COMPUTE, 3D_FULL_SCREEN, CUSTOM) and `power_dpm_force_performance_level`, each against the same short-job, long-job and memory-latency rows | the default profile starts work at half clock and lets the memory clock idle; the warmer fixes the first at a cost and the stable pstate fixes the second at a cost. A profile may fix both **[gated: root; a udev rule could hand the two files to this account]** |
| E2 | **Does the clock ramp differ with the screen off?** the same `clkrun.py` rows, screen on and off | the ramp was sampled with the screen on, the memory clock with it off; neither both ways |
| E3 | **Why the warmer slows staged uploads from 256 MiB** (9.7 vs 13.1 GB/s) when the GPU-side copy and direct writes do not move | decides whether the warmer can stay on while data streams over the link |
| E4 | **Pool round trip, two modes** (~120 and ~150-160 ns, same cores, same clock): count cache-line transfers and machine clears per round trip with `perf stat` in each mode | the only CPU row still behind Windows; a pool that stays in the fast mode gains 20% on tiny tasks |
| E5 | **RADV by version**: the loop header (`s_wait_alu`) and the descriptor load inside loops, on the current Mesa and on Mesa main | if a newer RADV hoists them, rule 17 becomes a note; if not, it is a report for Mesa with `chase.comp` as the test case |
| E6 | **Memory-weight network kernels with the weights copied into locals by hand** only for the rows a lane needs | what AMD's compiler does by itself (165 VGPRs at W=64); would show whether `mem` can match Windows on RADV without the full `reg` layout |

A4 (large pages) is answered for Linux by rule 15; on Windows it stays gated.

## Suggested order [DERIVED]

1. C1, C2, C3: cheap, and they make every later number more trustworthy.
2. A1: answers the portability question before anything is shipped.
3. B1 + D1 together: the largest untapped speed on this machine (matrix units), measured with its error.
4. B2, B3: attack the submit and transfer costs that set most crossovers.
5. A2, A6, D2: the CPU side for zarbor.

Results go into `measurements.md`; rules that come out of them go into `speed-playbook.md`.
