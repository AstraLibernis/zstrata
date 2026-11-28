# zstrata plan

**Owner (2026-10-15):** its own project for understanding: expand the GPU probe with RAM and CPU
probes to see the layers, levels and proper grouping, so that slow/fast and simple/complex work
lands in the right place (zarbor and the Vulkan network work were built without such a map).
Scope chosen: map + crossovers.

## S0 Skeleton ✓
One binary, `zstrata <cpu|link|gpu|cross|all>`, one CSV schema (`src/report.zig`), tests run
every probe at small sizes on the real GPU, ThreadSanitizer over the CPU tests, zig-precommit.

## S1 CPU map ✓
alu (width × chains, threads incl. SMT), chase, stream (1 thread, all threads), branch, sync,
alloc. Exact instructions checked in the disassembly.

## S2 Link ✓
upload (staged / direct-write / GPU copy), download (staged / direct-read), roundtrip.

## S3 GPU map ✓
Ported from the earlier Vulkan lab project (same probes, same fixes).

## S4 Crossovers ✓
map, nets, trees, chains, chain-long; GPU results checked against the CPU at every size
(mutation-checked).

## S5 Page and placement guide ✓
`docs/page/strata.html`, published as a private Artifact: CPU and GPU ladders side by side with the link between, crossover charts, rules for where work
and data go.

## S6 Native Windows ✓ (2026-11-08)
Same source on Windows 11 (AMD driver): `src/cpu/host.zig` for the OS calls, Vulkan loaded by
hand (Zig 0.17's DynLib has no Windows), shared-memory sizes read from the driver (32 KiB here,
64 on RADV). 4 runs before the fixes (run7-10), 4 after (run11-14), 4 with the speed
experiments (run15-18). Every crossover verdict is the same as on Linux.

## S7 Standalone + Windows gaps ✓ (2026-11-08)
**Owner:** zstrata is the hardware benchmark others are checked against, so no project
dependencies: own pool (`src/cpu/pool.zig`), zarbor removed. Each Windows deficit tested for a
cause in our code first (measurements.md, "Windows gaps: ours or the OS?").

## S8 Speed experiments + playbook ✓ (2026-11-08)
**Owner:** experiment with the ways to keep things fast and document them for the other
projects. New rows: hybrid fence wait, CPU cost of each wait, batching dispatches, vec4 loads
from shared memory (`lds4`) and from buffers (`mem4`), spawn with a 1 MiB stack. Results and
how to apply them: `docs/speed-playbook.md`.

## S9 Native Linux, gaps against Windows ✓ (2026-11-26)
**Owner:** test on Linux; find, prove and fix what trails Windows. 4 runs before (run29-32),
4 after with the warmer (run33-36), 4 after without (run37-40): aligned 2 MiB-page buffers (`lib/mem.zig`), the GPU keep-warm job
(`--gpu-warm`, `gpu/warm.zig`), loops unrolled in `chains`, `chase` and `fma` shaders.
`measurements.md`, "Linux gaps"; playbook rules 14-18. Open:
- Windows rerun with the new shaders (alu, lanes, occupancy, gpu chase, chains GPU rows)
- a clean Linux set with the screen on (runs 33-40 had it off: GPU memory rows from 96 MiB)
- GPU power settings (the sandbox may write them since 2026-11-27): first pass in
  `measurements.md`. A custom profile with GFXCLK min active 3300 replaces the warmer in single
  runs. Next: 4 full runs under it; core clock and memory clock both up (custom FCLK line,
  or other custom values); the same with the screen locked; wrap set/restore into the run
- pool round trip, workers awake: 1.1-1.2x behind, bimodal, cause not found
- `mem` / `mem4` / `lds4` network kernels on RADV; sustained GPU peaks 2-4% lower
- the page (`docs/page/strata.html`) and `docs/data/strata.csv` still show the 2026-10-15
  container run

## Ceiling target (Linux against Windows, set 2026-11-27)
The target is **parity with Windows on the same machine, not beating it**. Rows where Linux is
ahead are never work items. Check: `tools/linux/ceiling.py <4 Windows runs> -- <4 Linux runs>`
(today: runs 23-26 against 41-44); exit 0 = met.

- **Met when**, with 4 runs per side: no probe/variant (3+ rows) with a median Windows advantage
  over 1.05x, and no row over 1.10x whose runs do not overlap. The exception is a gap listed in
  `docs/ceiling-exceptions.tsv` whose cause lies outside this code (`power`, `driver`,
  `permission`) or that is `tried`.
- **Each open gap gets one bounded experiment** (one named hypothesis, one A/B, a
  `measurements.md` entry). If it does not close the gap, list it as `tried` and stop. No second
  round unless new evidence names a new cause.
- **A change counts** only when it beats the noise in repeat runs (non-overlapping ranges over
  4 runs) and has a named cause. A gain from one run, or with no mechanism, is not kept.
- **Not work items:** rows ahead of Windows, gaps inside run-to-run noise, and the absolute
  hardware limits this project measures (those are the map, not targets).

Open items on 2026-11-27 (runs 41-44), each with its one experiment:
| gap | Windows ahead | experiment | if it fails |
|---|---|---|---|
| `gpu/chase` past the caches | 2.3-3.1x | none in code: the memory clock drops with the screen off; needs `pp_dpm_mclk` (owner) | listed as `permission` |
| `gpu/mlp` mem / mem4 / lds4 / reg-one | 1.07-1.6x | E6, weights copied into locals by hand | `tried` (RADV, report to Mesa via E5) |
| `cpu/sync/pool-hot` | 1.16x | E4, `perf stat` of both modes | `tried` |
| `link/download` staged / direct-read, `link/upload/staged` | 1.14-1.15x | one `perf` / timestamp split of a 256 MiB download | `tried` |
| single rows: `cross/map` cpu-1 and gpu-with-transfer, `cpu/chase/latency` | 1 row each | the next 4-run set; open only if they stay | dropped as noise |

When the table is empty or all `tried` / outside the code, S9 is finished and Linux work stops.

## Later (not started)
- Apply the playbook to the other projects (zarbor, zmural and others), one at a time, when
  each is being worked on
- Apply the guide to real code: M9's training kernel (register spill), zarbor (which parts, if
  any, belong on the GPU)
- Repeat on another day (4 same-day runs agree: docs/measurements.md, Repeatability)
- New experiments (speed and accuracy, hardware-specific): [`experiments.md`](experiments.md).
  Stopping point 2026-11-08 night: C1, C2, C3 and A1 done (09a927f, 0bbacce). Next: B1 + D1
  together (MLP layers on the matrix units, with their error against an fp64 reference). Open
  question from C2: why the launch phase is bimodal (~2x) between runs.
