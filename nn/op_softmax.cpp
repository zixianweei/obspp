#include "nn/op_softmax.h"

#include "base/context.h"
#include "base/logger.h"
#include "base/macros.h"
#include "base/tensor.h"
#include "nn/types.h"

namespace cutenn {

int normalize_axis(int axis, int ndims) {
  return (axis < 0) ? axis + ndims : axis;
}

void make_attribute_dims_1(OpSoftmaxAttribute &attr, const TShape &ishape,
                           const TShape &oshape, int axis) {
  CUTENN_CHECK(ishape.size() == 1U, "{}: invalid dims = {}.", __func__,
               ishape.size());
  axis = normalize_axis(axis, ishape.size());
  attr.axis = axis;
  attr.i_h = 1;
  attr.i_w = ishape[0];
  attr.o_h = 1;
  attr.o_w = oshape[0];
}

void make_attribute_dims_2(OpSoftmaxAttribute &attr, const TShape &ishape,
                           const TShape &oshape, int axis) {
  CUTENN_CHECK(ishape.size() == 2U, "{}: invalid dims = {}.", __func__,
               ishape.size());
  axis = normalize_axis(axis, ishape.size());
  attr.axis = axis;
  attr.i_h = ishape[1];
  attr.i_w = ishape[0];
  attr.o_h = oshape[1];
  attr.o_w = oshape[0];
}

void make_attribute_dims_3(OpSoftmaxAttribute &attr, const TShape &ishape,
                           const TShape &oshape, int axis) {
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

void make_attribute_dims_4(OpSoftmaxAttribute &attr, const TShape &ishape,
                           const TShape &oshape, int axis) {
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

bool OpSoftmax::Forward(Tensor &src, Tensor &dst) {
  CUTENN_CHECK(src.GetDims() == dst.GetDims(),
               "{}: src dims = {}, dst dims = {}.", __func__, src.GetDims(),
               dst.GetDims());

  MTLComputeCommandEncoderPtr commandEncoder =
      Context::GetInstance().GetCommandEncoder();
  if (commandEncoder == nullptr) {
    CUTENN_LOG_ERROR("{}: command encoder is nil", __func__);
    return false;
  }

#if defined(CUTENN_METAL_DEBUG)
  Context::GetInstance().BeginCaptureScope();
  Context::GetInstance().PushCommandEncoderToDebugGroup(commandEncoder,
                                                        "OpSoftmax::Forward");
#endif

  MTLComputePipelineStatePtr computePipelineState =
      Context::GetInstance().GetComputePipelineState(
          GetKernelName(src.GetDims(), GetAxis()));
  if (computePipelineState == nullptr) {
    CUTENN_LOG_ERROR("{}: compute pipeline state is nil");
    return false;
  }

  Context::GetInstance().SetCommandEncoderComputePipelineState(
      commandEncoder, computePipelineState);

  Context::GetInstance().SetCommandEncoderBuffer(commandEncoder,
                                                 src.GetRawBuffer(), 0, 0);
  Context::GetInstance().SetCommandEncoderBuffer(commandEncoder,
                                                 dst.GetRawBuffer(), 0, 1);
  Context::GetInstance().SetCommandEncoderBuffer(commandEncoder,
                                                 MakeAttribute(src, dst), 0, 2);

  unsigned int maxTotalThreadsPerThreadgroup =
      Context::GetInstance().GetMaxTotalThreadsPerThreadgroup(
          computePipelineState);
  unsigned int threadExecutionWidth =
      Context::GetInstance().GetThreadExecutionWidth(computePipelineState);

#if defined(CUTENN_METAL_DEBUG)
  Context::GetInstance().PopCommandEncoderFromDebugGroup(commandEncoder);
#endif

  Context::GetInstance().EndEncoding(commandEncoder);

  if (!Context::GetInstance().Commit()) {
    CUTENN_LOG_ERROR("{}: commit failed", __func__);
    return false;
  }

#if defined(CUTENN_METAL_DEBUG)
  Context::GetInstance().EndCaptureScope();
#endif

  return true;
}

std::string OpSoftmax::GetKernelName(int dims, int axis) const {
  return "kernel_softmax_dims_" + std::to_string(dims) + "_axis_" +
         std::to_string(axis);
}

MTLBufferPtr OpSoftmax::MakeAttribute(Tensor &src, Tensor &dst) {
  OpSoftmaxAttribute attribute;
  switch (src.GetDims()) {
  case 1:
    make_attribute_dims_1(attribute, src.GetShape(), dst.GetShape(), GetAxis());
    break;
  case 2:
    make_attribute_dims_2(attribute, src.GetShape(), dst.GetShape(), GetAxis());
    break;
  case 3:
    make_attribute_dims_3(attribute, src.GetShape(), dst.GetShape(), GetAxis());
    break;
  case 4:
    make_attribute_dims_4(attribute, src.GetShape(), dst.GetShape(), GetAxis());
    break;
  default:
    CUTENN_LOG_CRITICAL("{}: dims = {} is not supported", __func__,
                        src.GetDims());
    break;
  }
  return Context::GetInstance().MakeBuffer(&attribute, sizeof(attribute));
}

} // namespace cutenn
