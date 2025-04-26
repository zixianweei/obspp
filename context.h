#ifndef CUTE_CONTEXT_H_
#define CUTE_CONTEXT_H_

#include <string>
#include <vector>

#ifdef __OBJC__
@class CuteContextImpl;
#else
typedef struct objc_object CuteContextImpl;
#endif

#ifdef __OBJC__
#    include <Foundation/Foundation.h>
#    include <Metal/Metal.h>
typedef id<MTLDevice> MTLDevicePtr;
typedef id<MTLCommandQueue> MTLCommandQueuePtr;
typedef id<MTLComputePipelineState> MTLComputePipelineStatePtr;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoderPtr;
#else
typedef void* MTLDevicePtr;
typedef void* MTLCommandQueuePtr;
typedef void* MTLComputePipelineStatePtr;
typedef void* MTLComputeCommandEncoderPtr;
#endif

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

    void BeginCapture();
    void EndCapture();

#ifdef __OBJC__
    MTLSize CalculateGridSize(const std::vector<int>& shape);
    MTLSize CalculateThreadgroupSize(const MTLSize& gridSize);
#endif

private:
    Context();
#ifdef __OBJC__
    __strong CuteContextImpl* impl_;
#else
    CuteContextImpl* impl_;
#endif
};

}  // namespace cute

#endif  // !CUTE_CONTEXT_H_
