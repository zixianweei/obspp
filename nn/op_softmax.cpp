#include "nn/op_softmax.hpp"

#include "base/context.hpp"
#include "base/logger.hpp"
#include "base/macros.hpp"
#include "base/tensor.hpp"
#include "nn/types.hpp"

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

  MTLComputePipelineStatePtr computePipelineState =
      Context::GetInstance().GetComputePipelineState(GetKernelName());
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

  if (!Context::GetInstance().Commit()) {
    CUTENN_LOG_ERROR("{}: commit failed", __func__);
    return false;
  }

  return true;
}

std::string OpSoftmax::GetKernelName() const {
  int axis = GetAxis();
  switch (axis) {
  case 0:
    return "kernel_softmax_axis_0";
  case 1:
    return "kernel_softmax_axis_1";
  case 2:
    return "kernel_softmax_axis_2";
  case 3:
    return "kernel_softmax_axis_3";
  default:
    break;
  }
  CUTENN_LOG_CRITICAL("{}: axis {} is not supported", __func__, axis);
  return "";
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
