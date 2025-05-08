#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

kernel void kernel_softmax_dims_1_axis_0(
    const device float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant cutenn::OpSoftmaxAttribute& attr [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (any(gid >= uint3(attr.o_w, 1, 1))) {
        return;
    }
}

kernel void kernel_softmax_dims_2_axis_0()
{}

kernel void kernel_softmax_dims_2_axis_1()
{}

kernel void kernel_softmax_dims_3_axis_0()
{}

kernel void kernel_softmax_dims_3_axis_1()
{}

kernel void kernel_softmax_dims_3_axis_2()
{}

kernel void kernel_softmax_dims_4_axis_0()
{}

kernel void kernel_softmax_dims_4_axis_1()
{}

kernel void kernel_softmax_dims_4_axis_2()
{}

kernel void kernel_softmax_dims_4_axis_3()
{}