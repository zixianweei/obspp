#ifndef CUTE_CONTEXT_H_
#define CUTE_CONTEXT_H_

#include <string>
#include <vector>

#include "macros.h"

CUTE_OBJC_FORWARD_DECLARATION(CuteContextImpl);

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#endif

CUTE_TYPE_ALIAS(id<MTLDevice>, MTLDevicePtr);
CUTE_TYPE_ALIAS(id<MTLCommandQueue>, MTLCommandQueuePtr);
CUTE_TYPE_ALIAS(id<MTLComputePipelineState>, MTLComputePipelineStatePtr);
CUTE_TYPE_ALIAS(id<MTLComputeCommandEncoder>, MTLComputeCommandEncoderPtr);

namespace cute {

class Context
{
    struct Impl;

public:
    static Context& GetInstance();
    
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept = delete;
    Context& operator=(Context&&) noexcept = delete;

    MTLDevicePtr GetDevice();
    MTLCommandQueuePtr GetCommandQueue();
    MTLComputePipelineStatePtr findComputePipelineState(const std::string& kname);
    MTLComputeCommandEncoderPtr GetCommandEncoder();
    bool Commit();

private:
    Context();

    CuteContextImpl* impl_;
};

}  // namespace cute

#endif  // !CUTE_CONTEXT_H_
