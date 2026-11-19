// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis
//
// Matrix-unit peak: each subgroup keeps one A tile, one B tile and ACC accumulators in
// registers and multiplies them ITERS times; no memory traffic in the loop. TA is the input
// type, TC the accumulator type; tiles are M x N x K (16 x 16 x 16 on this card).
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
layout(local_size_x = 256) in; // 4 subgroups of 64 (the host pins the width)
layout(push_constant) uniform Push { uint iters; } pc;
layout(set = 0, binding = 0) writeonly buffer Out { TC o[]; };
const uint ACC = 8;

void main() {
    coopmat<TA, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a = coopmat<TA, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>(TA(ONE));
    coopmat<TA, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b = coopmat<TA, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>(TA(ONE));
    coopmat<TC, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[ACC];
    [[unroll]] for (uint j = 0; j < ACC; j++) acc[j] = coopmat<TC, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(TC(j));
    for (uint it = 0; it < pc.iters; it++) {
        [[unroll]] for (uint j = 0; j < ACC; j++) acc[j] = coopMatMulAdd(a, b, acc[j]);
    }
    [[unroll]] for (uint j = 1; j < ACC; j++) acc[0] = acc[0] + acc[j];
    uint sg = gl_WorkGroupID.x * 4 + gl_LocalInvocationID.x / 64;
    coopMatStore(acc[0], o, sg * 256, 16, gl_CooperativeMatrixLayoutRowMajor);
}
