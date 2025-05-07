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

  int naxis = normalize_axis(GetAxis(), src.GetDims());
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
  // attribute.i_shape = src.GetShape();
  // attribute.o_shape = dst.GetShape();
  attribute.axis = GetAxis();
  return Context::GetInstance().MakeBuffer(&attribute, sizeof(attribute));
}

} // namespace cutenn
