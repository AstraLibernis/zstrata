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

## Later (not started)
- Run the new rows on Linux (lds4, mem4, wait, batch, spawn-join-1mib, roundtrip splits have
  Windows numbers only)
- Apply the playbook to the other projects (zarbor, zmural and others), one at a time, when
  each is being worked on
- Apply the guide to real code: M9's training kernel (register spill), zarbor (which parts, if
  any, belong on the GPU)
- Repeat on another day (4 same-day runs agree: docs/measurements.md, Repeatability)
- New experiments (speed and accuracy, hardware-specific): [`experiments.md`](experiments.md).
  Stopping point 2026-11-08 night: C1, C2, C3 and A1 done (09a927f, 0bbacce). Next: B1 + D1
  together (MLP layers on the matrix units, with their error against an fp64 reference). Open
  question from C2: why the launch phase is bimodal (~2x) between runs.
