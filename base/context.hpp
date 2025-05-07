#ifndef CUTENN_BASE_CONTEXT_H_
#define CUTENN_BASE_CONTEXT_H_

#include <string>

#include "base/macros.hpp"

CUTENN_OBJC_FORWARD_DECLARATION(CuteContextImpl);

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#endif

CUTENN_TYPE_ALIAS(id<MTLDevice>, MTLDevicePtr);
CUTENN_TYPE_ALIAS(id<MTLCommandQueue>, MTLCommandQueuePtr);
CUTENN_TYPE_ALIAS(id<MTLComputePipelineState>, MTLComputePipelineStatePtr);
CUTENN_TYPE_ALIAS(id<MTLComputeCommandEncoder>, MTLComputeCommandEncoderPtr);

namespace cutenn {

class Context {
  struct Impl;

public:
  static Context &GetInstance();

  ~Context();

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) noexcept = delete;
  Context &operator=(Context &&) noexcept = delete;

  MTLDevicePtr GetDevice();
  MTLCommandQueuePtr GetCommandQueue();
  MTLComputePipelineStatePtr findComputePipelineState(const std::string &kname);
  MTLComputeCommandEncoderPtr GetCommandEncoder();
  bool Commit();

private:
  Context();

  CuteContextImpl *impl_;
};

} // namespace cutenn

#endif // !CUTENN_BASE_CONTEXT_H_
