#include "op_flip.h"

#include "context.h"
#include "logger.h"
#include "tensor.h"

namespace cute {

bool OpFlip::Forward(Tensor& src, Tensor& dst)
{
    MTLComputeCommandEncoderPtr commandEncoder = Context::GetInstance().GetCommandEncoder();
    if (commandEncoder == nil) {
        CUTE_LOG_ERROR("{}: command encoder is nil", __func__);
        return false;
    }

    MTLComputePipelineStatePtr computePipelineState = Context::GetInstance().findComputePipelineState(GetKernelName());
    if (computePipelineState == nil) {
        CUTE_LOG_ERROR("{}: compute pipeline state is nil");
        return false;
    }

    Context::GetInstance().BeginCapture();

    [commandEncoder setComputePipelineState:computePipelineState];

    [commandEncoder setBuffer:src.GetRawBuffer() offset:0 atIndex:0];
    [commandEncoder setBuffer:dst.GetRawBuffer() offset:0 atIndex:1];

    NSUInteger maxTotalThreadsPerThreadgroup = [computePipelineState maxTotalThreadsPerThreadgroup];
    NSUInteger threadExecutionWidth = [computePipelineState threadExecutionWidth];
    NSUInteger threadExecutionHeight = maxTotalThreadsPerThreadgroup / threadExecutionWidth;

    MTLSize threadPerThreadgroup = MTLSizeMake(threadExecutionWidth, threadExecutionHeight, 1);
    MTLSize threads = MTLSizeMake(src.Shape()[3], src.Shape()[2], 1);

    [commandEncoder dispatchThreads:threads threadsPerThreadgroup:threadPerThreadgroup];
    [commandEncoder endEncoding];

    if (!Context::GetInstance().Commit()) {
        CUTE_LOG_ERROR("{}: commit failed", __func__);
        return false;
    }

    Context::GetInstance().EndCapture();

    return true;
}

std::string OpFlip::GetKernelName() const
{
    return "kernel_flip";
}

} // namespace cute
