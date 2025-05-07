#include <metal_stdlib>
#include "nn/property.h"
using namespace metal;

kernel void kernel_flip(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant cute::TensorProperty& property [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    auto outptr = output + (property.o_h - 1 - gid.y) * property.o_w * property.o_c + (gid.x) * property.o_c;
    auto inptr = input + (gid.y) * property.i_w * property.i_c + (gid.x) * property.i_c;

    *outptr = *inptr;
    *(outptr + 1) = *(inptr + 1);
    *(outptr + 2) = *(inptr + 2);
}
