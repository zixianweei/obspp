#include "op_flip.h"

#include "base/context.h"
#include "base/logger.h"
#include "base/tensor.h"
#include "nn/types.h"

static id<MTLBuffer> make_attribute(const cutenn::Tensor &src,
                                    const cutenn::Tensor &dst) {
  cutenn::OpFlipAttribute attribute;
  attribute.i_h = src.GetShape()[1];
  attribute.i_w = src.GetShape()[0];
  attribute.o_h = dst.GetShape()[1];
  attribute.o_w = dst.GetShape()[0];
  id<MTLBuffer> buffer =
      [[CUTE_CONTEXT device] newBufferWithBytes:&attribute
                                         length:sizeof(cutenn::OpFlipAttribute)
                                        options:MTLResourceStorageModeShared];
  if (buffer == nullptr) {
    CUTENN_LOG_ERROR("{}: failed to allocate attribute buffer.", __func__);
    return nil;
  }
  return buffer;
}

namespace cutenn {

bool OpFlip::Forward(Tensor &src, Tensor &dst) {
  CUTENN_CHECK(src.GetDims() == dst.GetDims(),
               "{}: src and dst must have the same dims", __func__)
  CUTENN_CHECK(src.GetDims() == 2, "{}: src and dst must have 2 dims", __func__)

  id<MTLComputeCommandEncoder> commandEncoder = [CUTE_CONTEXT createEncoder];
  if (commandEncoder == nullptr) {
    CUTENN_LOG_ERROR("{}: command encoder is nil", __func__);
    return false;
  }

#if defined(CUTENN_METAL_DEBUG)
  [CUTE_CONTEXT beginCapture];
  [commandEncoder pushDebugGroup:[[NSString alloc] initWithFormat:@"OpFlip"]];
#endif

  NSString *kernelName =
      [NSString stringWithFormat:@"%s", GetKernelName().c_str()];
  id<MTLComputePipelineState> computePipelineState =
      [CUTE_CONTEXT findComputePipelineState:kernelName];
  if (computePipelineState == nullptr) {
    CUTENN_LOG_ERROR("{}: compute pipeline state is nil");
    return false;
  }

  [commandEncoder setComputePipelineState:computePipelineState];
  [commandEncoder setBuffer:src.GetRawBuffer() offset:0 atIndex:0];
  [commandEncoder setBuffer:dst.GetRawBuffer() offset:0 atIndex:1];
  [commandEncoder setBuffer:make_attribute(src, dst) offset:0 atIndex:2];

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

std::string OpFlip::GetKernelName() const { return "kernel_flip"; }

} // namespace cutenn
