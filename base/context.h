#ifndef CUTENN_BASE_CONTEXT_H_
#define CUTENN_BASE_CONTEXT_H_

#include <string>

#include "base/macros.h"
#include "base/types.h"

#ifdef __OBJC__

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

typedef NSMutableDictionary<NSString *, id<MTLComputePipelineState>>
    *ComputePipelineStateDictionary;
typedef NSMutableArray<id<MTLCommandBuffer>> *CommandBufferArray;

@interface MTL4CuteContext : NSObject
@property(strong, nonatomic) id<MTLDevice> device;
@property(strong, nonatomic) id<MTLCommandQueue> commandQueue;
@property(strong, nonatomic) id<MTLLibrary> library;
@property(assign, nonatomic) BOOL hasSimdGroupReduction;
@property(assign, nonatomic) BOOL hasBFloat;
@property(strong, nonatomic) dispatch_queue_t queue;
@property(strong, nonatomic) ComputePipelineStateDictionary cachedCPS;
@property(strong, nonatomic) id<MTLCommandBuffer> commandBuffer;
@property(strong, nonatomic) CommandBufferArray schedCommandBuffer;
#if defined(CUTENN_METAL_DEBUG)
@property(strong, nonatomic) id<MTLCaptureScope> captureScope;
@property(assign, nonatomic) BOOL isCaptureScopeOn;
#endif // CUTENN_METAL_DEBUG

- (instancetype)init;
- (void)dealloc;
- (id<MTLComputePipelineState>)findComputePipelineState:(NSString *)kernelName;
- (id<MTLComputeCommandEncoder>)createEncoder;
- (BOOL)commit;
#if defined(CUTENN_METAL_DEBUG)
- (void)makeCaptureScopeAvailable;
- (void)beginCapture;
- (void)endCapture;
#endif // CUTENN_METAL_DEBUG
@end

#else
#error DO NOT USE THIS FILE IN NON-OBJC BUILDS.
#endif // __OBJC__

namespace cutenn {

class ContextOwner {
public:
  static ContextOwner &GetInstance();
  ~ContextOwner();

  ContextOwner(const ContextOwner &) = delete;
  ContextOwner &operator=(const ContextOwner &) = delete;
  ContextOwner(ContextOwner &&) = delete;
  ContextOwner &operator=(ContextOwner &&) = delete;

  MTL4CuteContext *Get() { return context_; }

private:
  ContextOwner();

  MTL4CuteContext *context_;
};

} // namespace cutenn

#define CUTE_CONTEXT cutenn::ContextOwner::GetInstance().Get()

#endif // !CUTENN_BASE_CONTEXT_H_
