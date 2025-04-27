#include <metal_stdlib>
using namespace metal;

kernel void kernel_grayscale(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant cute::TensorProperty& property [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // if (gid.x < 0 || gid.x >= 640 * 3 || gid.y < 0 || gid.y >= 360) {
    //     return;
    // }

    auto outptr = output + gid.y * property.o_h + (gid.x);
    auto inptr = input + (gid.y) * property.i_h * property.i_c + (gid.x) * property.i_c;

    *outptr = *inptr;
    *outptr = *outptr + *(inptr + 1);
    *outptr = *outptr + *(inptr + 2);
    *outptr = *outptr / 3.0;
}
