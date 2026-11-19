// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 AstraLibernis
//
// The combined probe: a population of square networks (width W, the same W x W layer applied
// LAYERS times, tanh after each) run on ITERS inputs each. One workgroup = one 64-lane
// subgroup holding NPS networks; each network gets 64/NPS lanes, and a lane computes the
// neurons j, j + lanes, j + 2*lanes, ... of its network. Where the weights live is the
// variable: PLACE_REG (copied once into each lane's registers), PLACE_LDS (copied once into
// shared memory) or PLACE_MEM (read from the buffer on every use, through the caches).
// Activations always sit in shared memory. Inputs are made from the indices, so the only
// memory traffic is weights.
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x = 64) in;
layout(constant_id = 0) const uint W = 64;
layout(constant_id = 1) const uint NPS = 1;
const uint LPN = 64 / NPS;                    // lanes per network
const uint R = (W + LPN - 1) / LPN;           // neurons per lane
layout(push_constant) uniform Push { uint iters; uint layers; } pc;
#if defined(MEM_VEC4)
// The same weights, read from the buffer as aligned vec4s (16-byte loads). Needs W % 4 == 0.
layout(set = 0, binding = 0) readonly buffer Weights { vec4 w4[]; };
#else
layout(set = 0, binding = 0) readonly buffer Weights { float w[]; };
#endif
layout(set = 0, binding = 1) writeonly buffer Out { float o[]; };
#if defined(LDS_VEC4) || defined(MEM_VEC4)
// Same layout, read 16 bytes at a time: every load is a whole aligned vec4, so the compiler must
// emit one wide shared-memory load where it might otherwise issue two-float loads. Needs W % 4 == 0.
shared vec4 x4[NPS * W / 4];
#define XGET(i) x4[(i) >> 2][(i) & 3u]
#define XSET(i, v) x4[(i) >> 2][(i) & 3u] = (v)
#else
shared float x[NPS * W];
#define XGET(i) x[i]
#define XSET(i, v) x[i] = (v)
#endif

#if defined(LDS_VEC4)
shared vec4 ws4[NPS * W * W / 4];
#elif defined(PLACE_LDS)
shared float ws[NPS * W * W];
#define WT(r, row, k) ws[nl * W * W + (row) * W + (k)]
#elif defined(PLACE_REG)
#define WT(r, row, k) wr[(r) * W + (k)]
#else
#define WT(r, row, k) w[wo + (row) * W + (k)]
#endif

// Input k of input number n, in [-0.5, 0.5).
float inputValue(uint n, uint k) {
    return float((n * 2654435761u + k * 40503u) >> 24) * (1.0 / 256.0) - 0.5;
}

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint nl = lane / LPN, j = lane % LPN;
    uint net = gl_WorkGroupID.x * NPS + nl;
    uint wo = net * W * W;
#if defined(LDS_VEC4)
    uint base = gl_WorkGroupID.x * NPS * W * W;
    for (uint i = lane; i < NPS * W * W / 4; i += 64)
        ws4[i] = vec4(w[base + 4 * i], w[base + 4 * i + 1], w[base + 4 * i + 2], w[base + 4 * i + 3]);
#elif defined(PLACE_LDS)
    uint base = gl_WorkGroupID.x * NPS * W * W;
    for (uint i = lane; i < NPS * W * W; i += 64) ws[i] = w[base + i];
#elif defined(PLACE_REG)
    float wr[R * W];
    [[unroll]] for (uint r = 0; r < R; r++) {
        uint row = min(j + r * LPN, W - 1);
        [[unroll]] for (uint k = 0; k < W; k++) wr[r * W + k] = w[wo + row * W + k];
    }
#endif
    float total = 0.0;
    for (uint n = 0; n < pc.iters; n++) {
        barrier();
        for (uint k = j; k < W; k += LPN) XSET(nl * W + k, inputValue(n, k));
        barrier();
        for (uint l = 0; l < pc.layers; l++) {
            float y[R];
            [[unroll]] for (uint r = 0; r < R; r++) {
                uint row = min(j + r * LPN, W - 1); // lanes past the last neuron redo it
                float acc = 0.0;
#if defined(LDS_VEC4)
                // The same fma chain in the same order as below, four terms per load.
                [[unroll]] for (uint k4 = 0; k4 < W / 4; k4++) {
                    vec4 wv = ws4[(nl * W * W + row * W) / 4 + k4];
                    vec4 xv = x4[nl * W / 4 + k4];
                    acc = fma(wv.x, xv.x, acc);
                    acc = fma(wv.y, xv.y, acc);
                    acc = fma(wv.z, xv.z, acc);
                    acc = fma(wv.w, xv.w, acc);
                }
#elif defined(MEM_VEC4)
                [[unroll]] for (uint k4 = 0; k4 < W / 4; k4++) {
                    vec4 wv = w4[(wo + row * W) / 4 + k4];
                    vec4 xv = x4[nl * W / 4 + k4];
                    acc = fma(wv.x, xv.x, acc);
                    acc = fma(wv.y, xv.y, acc);
                    acc = fma(wv.z, xv.z, acc);
                    acc = fma(wv.w, xv.w, acc);
                }
#else
                [[unroll]] for (uint k = 0; k < W; k++) acc = fma(WT(r, row, k), XGET(nl * W + k), acc);
#endif
                y[r] = tanh(acc);
            }
            barrier();
            [[unroll]] for (uint r = 0; r < R; r++) {
                uint row = j + r * LPN;
                if (row < W) XSET(nl * W + row, y[r]);
            }
            barrier();
        }
        if (j == 0) total += XGET(nl * W);
    }
    if (j == 0) o[net] = total;
}
