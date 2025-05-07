#include "op_flip.h"

#include "base/context.hpp"
#include "base/logger.hpp"
#include "base/tensor.hpp"

namespace cutenn {

bool OpFlip::Forward(Tensor &src, Tensor &dst) {
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
  Context::GetInstance().SetCommandEncoderBuffer(
      commandEncoder, OpBase::AllocateTensorProperty(src, dst), 0, 2);

  unsigned int maxTotalThreadsPerThreadgroup =
      Context::GetInstance().GetMaxTotalThreadsPerThreadgroup(
          computePipelineState);
  unsigned int threadExecutionWidth =
      Context::GetInstance().GetThreadExecutionWidth(computePipelineState);

  unsigned int threadExecutionHeight =
      maxTotalThreadsPerThreadgroup / threadExecutionWidth;

  Size threadsPerThreadgroup =
      MakeSize(threadExecutionWidth, threadExecutionHeight, 1);
  Size threads = MakeSize(src.GetShape()[3], src.GetShape()[2], 1);

  Context::GetInstance().DispatchThreads(commandEncoder, threads,
                                         threadsPerThreadgroup);
  Context::GetInstance().EndEncoding(commandEncoder);

  if (!Context::GetInstance().Commit()) {
    CUTENN_LOG_ERROR("{}: commit failed", __func__);
    return false;
  }

  return true;
}

std::string OpFlip::GetKernelName() const { return "kernel_flip"; }

} // namespace cutenn
