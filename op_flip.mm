#include "op_flip.h"

#include "base/logger.h"
#include "context.h"
#include "tensor.h"

namespace cute {

bool OpFlip::Forward(Tensor &src, Tensor &dst) {
  MTLComputeCommandEncoderPtr commandEncoder =
      Context::GetInstance().GetCommandEncoder();
  if (commandEncoder == nil) {
    CUTENN_LOG_ERROR("{}: command encoder is nil", __func__);
    return false;
  }

  MTLComputePipelineStatePtr computePipelineState =
      Context::GetInstance().findComputePipelineState(GetKernelName());
  if (computePipelineState == nil) {
    CUTENN_LOG_ERROR("{}: compute pipeline state is nil");
    return false;
  }

  [commandEncoder setComputePipelineState:computePipelineState];

  [commandEncoder setBuffer:src.GetRawBuffer() offset:0 atIndex:0];
  [commandEncoder setBuffer:dst.GetRawBuffer() offset:0 atIndex:1];
  [commandEncoder setBuffer:OpBase::AllocateTensorProperty(src, dst)
                     offset:0
                    atIndex:2];

  NSUInteger maxTotalThreadsPerThreadgroup =
      [computePipelineState maxTotalThreadsPerThreadgroup];
  NSUInteger threadExecutionWidth = [computePipelineState threadExecutionWidth];
  NSUInteger threadExecutionHeight =
      maxTotalThreadsPerThreadgroup / threadExecutionWidth;

  MTLSize threadPerThreadgroup =
      MTLSizeMake(threadExecutionWidth, threadExecutionHeight, 1);
  MTLSize threads = MTLSizeMake(src.Shape()[3], src.Shape()[2], 1);

  [commandEncoder dispatchThreads:threads
            threadsPerThreadgroup:threadPerThreadgroup];
  [commandEncoder endEncoding];

  if (!Context::GetInstance().Commit()) {
    CUTENN_LOG_ERROR("{}: commit failed", __func__);
    return false;
  }

  return true;
}

std::string OpFlip::GetKernelName() const { return "kernel_flip"; }

} // namespace cute
