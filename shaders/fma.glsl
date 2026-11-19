// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis
//
// The ALU probe shared by fma32 and fma16: each lane keeps K independent values live and
// applies one fused multiply-add to each, ITERS times. Only the first LANES lanes of each
// subgroup work (the lane probe); SHARED floats of shared memory are reserved per workgroup
// (the occupancy probe: the driver fits fewer workgroups on a compute unit as it grows).
// K and SHARED are specialization constants; the host sweeps them.
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
layout(local_size_x = 256) in;
layout(constant_id = 0) const uint K = 8;
layout(constant_id = 1) const uint SHARED = 1;
layout(push_constant) uniform Push { uint iters; uint lanes; float a; float b; } pc;
layout(set = 0, binding = 0) writeonly buffer Out { float o[]; };
shared float lds[SHARED];

void main() {
    T r[K];
    float seed = float(gl_GlobalInvocationID.x) * 1e-6;
    [[unroll]] for (uint i = 0; i < K; i++) r[i] = T(seed + float(i) * 1e-3);
    if (gl_SubgroupInvocationID < pc.lanes) {
        T a = T(pc.a), b = T(pc.b);
        for (uint it = 0; it < pc.iters; it++) {
            [[unroll]] for (uint i = 0; i < K; i++) r[i] = fma(r[i], a, b);
        }
    }
    T s = T(0);
    [[unroll]] for (uint i = 0; i < K; i++) s += r[i];
    // Write every element of the shared array: if only some were used, the compiler would
    // shrink the array to those and the workgroup would reserve less than SHARED floats.
    for (uint i = gl_LocalInvocationID.x; i < SHARED; i += 256) lds[i] = SUM(s) + float(i);
    barrier();
    o[gl_GlobalInvocationID.x] = lds[(gl_LocalInvocationID.x * 37u + 1u) % SHARED];
}
