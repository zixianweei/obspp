#include <metal_stdlib>
#include "nn/types.h"
using namespace metal;

kernel void kernel_flip(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant cutenn::OpFlipAttribute& attribute [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint i_idx = attribute.i_w * gid.y + gid.x;
    uint o_idx = attribute.o_w * gid.y + gid.x;
    output[o_idx] = input[i_idx];
}
