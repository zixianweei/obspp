#include "op_flip.h"

#include "base/context.hpp"
#include "base/logger.hpp"
#include "base/tensor.hpp"
#include "nn/types.hpp"

namespace cutenn {

bool OpFlip::Forward(Tensor &src, Tensor &dst) {
  CUTENN_CHECK(src.GetDims() == dst.GetDims(),
               "{}: src and dst must have the same dims", __func__)
  CUTENN_CHECK(src.GetDims() == 2, "{}: src and dst must have 2 dims", __func__)

  MTLComputeCommandEncoderPtr commandEncoder =
      Context::GetInstance().GetCommandEncoder();
  if (commandEncoder == nullptr) {
    CUTENN_LOG_ERROR("{}: command encoder is nil", __func__);
    return false;
  }

#if defined(CUTENN_METAL_DEBUG)
  Context::GetInstance().BeginCaptureScope();
  Context::GetInstance().PushCommandEncoderToDebugGroup(commandEncoder,
                                                        "OpFlip::Forward");
#endif

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

  unsigned int threadExecutionHeight =
      maxTotalThreadsPerThreadgroup / threadExecutionWidth;

  Size threadsPerThreadgroup =
      MakeSize(threadExecutionWidth, threadExecutionHeight, 1);
  Size threads = MakeSize(src.GetShape()[0], src.GetShape()[1], 1);

  Context::GetInstance().DispatchThreads(commandEncoder, threads,
                                         threadsPerThreadgroup);

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

std::string OpFlip::GetKernelName() const { return "kernel_flip"; }

MTLBufferPtr OpFlip::MakeAttribute(Tensor &src, Tensor &dst) {
  OpFlipAttribute attribute;
  attribute.i_h = src.GetShape()[1];
  attribute.i_w = src.GetShape()[0];
  attribute.o_h = dst.GetShape()[1];
  attribute.o_w = dst.GetShape()[0];
  return Context::GetInstance().MakeBuffer(&attribute, sizeof(OpFlipAttribute));
}

} // namespace cutenn
