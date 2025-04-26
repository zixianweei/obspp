#include <metal_stdlib>
using namespace metal;

kernel void kernel_flip(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // if (gid.x < 0 || gid.x >= 640 * 3 || gid.y < 0 || gid.y >= 360) {
    //     return;
    // }

    auto outptr = output + (359 - gid.y) * 640 * 3 + (gid.x) * 3;
    auto inptr = input + (gid.y) * 640 * 3 + (gid.x) * 3;

    *outptr = *inptr;
    *(outptr + 1) = *(inptr + 1);
    *(outptr + 2) = *(inptr + 2);
}
