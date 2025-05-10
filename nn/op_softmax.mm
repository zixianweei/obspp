#include "nn/op_softmax.h"

#include "base/context.h"
#include "base/logger.h"
#include "base/macros.h"
#include "base/tensor.h"
#include "nn/types.h"

int normalize_axis(int axis, int ndims) {
  return (axis < 0) ? axis + ndims : axis;
}

void make_attribute_dims_1(cutenn::OpSoftmaxAttribute &attr,
                           const cutenn::TShape &ishape,
                           const cutenn::TShape &oshape, int axis) {
  CUTENN_CHECK(ishape.size() == 1U, "{}: invalid dims = {}.", __func__,
               ishape.size());
  axis = normalize_axis(axis, ishape.size());
  attr.axis = axis;
  attr.i_h = 1;
  attr.i_w = ishape[0];
  attr.o_h = 1;
  attr.o_w = oshape[0];
}

void make_attribute_dims_2(cutenn::OpSoftmaxAttribute &attr,
                           const cutenn::TShape &ishape,
                           const cutenn::TShape &oshape, int axis) {
  CUTENN_CHECK(ishape.size() == 2U, "{}: invalid dims = {}.", __func__,
               ishape.size());
  axis = normalize_axis(axis, ishape.size());
  attr.axis = axis;
  attr.i_h = ishape[1];
  attr.i_w = ishape[0];
  attr.o_h = oshape[1];
  attr.o_w = oshape[0];
}

void make_attribute_dims_3(cutenn::OpSoftmaxAttribute &attr,
                           const cutenn::TShape &ishape,
                           const cutenn::TShape &oshape, int axis) {
  CUTENN_CHECK(ishape.size() == 3U, "{}: invalid dims = {}.", __func__,
               ishape.size());
  axis = normalize_axis(axis, ishape.size());
  attr.axis = axis;
  attr.i_c = ishape[2];
  attr.i_h = ishape[1];
  attr.i_w = ishape[0];
  attr.o_c = oshape[2];
  attr.o_h = oshape[1];
  attr.o_w = oshape[0];
}

void make_attribute_dims_4(cutenn::OpSoftmaxAttribute &attr,
                           const cutenn::TShape &ishape,
                           const cutenn::TShape &oshape, int axis) {
  CUTENN_CHECK(ishape.size() == 4U, "{}: invalid dims = {}.", __func__,
               ishape.size());
  axis = normalize_axis(axis, ishape.size());
  attr.axis = axis;
  attr.i_n = ishape[3];
  attr.i_c = ishape[2];
  attr.i_h = ishape[1];
  attr.i_w = ishape[0];
  attr.o_n = oshape[3];
  attr.o_c = oshape[2];
  attr.o_h = oshape[1];
  attr.o_w = oshape[0];
}

id<MTLBuffer> make_attribute(const cutenn::Tensor &src,
                             const cutenn::Tensor &dst, int axis) {
  cutenn::OpSoftmaxAttribute attribute;
  switch (src.GetDims()) {
  case 1:
    make_attribute_dims_1(attribute, src.GetShape(), dst.GetShape(), axis);
    break;
  case 2:
    make_attribute_dims_2(attribute, src.GetShape(), dst.GetShape(), axis);
    break;
  case 3:
    make_attribute_dims_3(attribute, src.GetShape(), dst.GetShape(), axis);
    break;
  case 4:
    make_attribute_dims_4(attribute, src.GetShape(), dst.GetShape(), axis);
    break;
  default:
    CUTENN_LOG_CRITICAL("{}: dims = {} is not supported", __func__,
                        src.GetDims());
    break;
  }

  id<MTLBuffer> buffer =
      [[CUTE_CONTEXT device] newBufferWithBytes:&attribute
                                         length:sizeof(attribute)
                                        options:MTLResourceStorageModeShared];
  if (buffer == nullptr) {
    CUTENN_LOG_ERROR("{}: failed to allocate attribute buffer.", __func__);
    return nil;
  }
  return buffer;
}

namespace cutenn {

bool OpSoftmax::Forward(Tensor &src, Tensor &dst) {
  CUTENN_CHECK(src.GetDims() == dst.GetDims(),
               "{}: src dims = {}, dst dims = {}.", __func__, src.GetDims(),
               dst.GetDims());

  id<MTLComputeCommandEncoder> commandEncoder = [CUTE_CONTEXT createEncoder];
  if (commandEncoder == nullptr) {
    CUTENN_LOG_ERROR("{}: command encoder is nil", __func__);
    return false;
  }

#if defined(CUTENN_METAL_DEBUG)
  [CUTE_CONTEXT beginCapture];
  [commandEncoder pushDebugGroup:[[NSString alloc] initWithFormat:@"OpFlip"]];
#endif

  NSString *kernelName = [NSString
      stringWithFormat:@"%s", GetKernelName(src.GetDims(), GetAxis()).c_str()];
  id<MTLComputePipelineState> computePipelineState =
      [CUTE_CONTEXT findComputePipelineState:kernelName];
  if (computePipelineState == nullptr) {
    CUTENN_LOG_ERROR("{}: compute pipeline state is nil");
    return false;
  }

  [commandEncoder setComputePipelineState:computePipelineState];
  [commandEncoder setBuffer:src.GetRawBuffer() offset:0 atIndex:0];
  [commandEncoder setBuffer:dst.GetRawBuffer() offset:0 atIndex:1];
  [commandEncoder setBuffer:make_attribute(src, dst, GetAxis())
                     offset:0
                    atIndex:2];

  NSUInteger maxTotalThreadsPerThreadgroup =
      [computePipelineState maxTotalThreadsPerThreadgroup];
  NSUInteger threadExecutionWidth = [computePipelineState threadExecutionWidth];
  NSUInteger threadExecutionHeight =
      maxTotalThreadsPerThreadgroup / threadExecutionWidth;

  MTLSize threadsPerThreadgroup =
      MTLSizeMake(threadExecutionWidth, threadExecutionHeight, 1);
  MTLSize threads = MTLSizeMake(src.GetShape()[0], src.GetShape()[1], 1);

  [commandEncoder dispatchThreads:threads
            threadsPerThreadgroup:threadsPerThreadgroup];

#if defined(CUTENN_METAL_DEBUG)
  [commandEncoder popDebugGroup];
#endif

  [commandEncoder endEncoding];

  if (![CUTE_CONTEXT commit]) {
    CUTENN_LOG_ERROR("{}: commit failed", __func__);
    return false;
  }

#if defined(CUTENN_METAL_DEBUG)
  [CUTE_CONTEXT endCapture];
#endif

  return true;
}

std::string OpSoftmax::GetKernelName(int dims, int axis) const {
  return "kernel_softmax_dims_" + std::to_string(dims) + "_axis_" +
         std::to_string(axis);
}

} // namespace cutenn
