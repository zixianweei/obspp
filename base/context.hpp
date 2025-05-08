#ifndef CUTENN_BASE_CONTEXT_H_
#define CUTENN_BASE_CONTEXT_H_

#include <string>

#include "base/macros.hpp"
#include "base/types.hpp"

CUTENN_OBJC_FORWARD_DECLARATION(CuteContextImpl);

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
  MTLComputePipelineStatePtr GetComputePipelineState(const std::string &kname);
  MTLComputeCommandEncoderPtr GetCommandEncoder();

  MTLBufferPtr MakeBuffer(const void *data, size_t size);
  void SetCommandEncoderComputePipelineState(void *encoder, void *state);
  void SetCommandEncoderBuffer(void *encoder, void *buffer, int offset,
                               int index);
  unsigned int GetMaxTotalThreadsPerThreadgroup(void *state);
  unsigned int GetThreadExecutionWidth(void *state);
  void DispatchThreads(void *encoder, const Size &threads,
                       const Size &threadsPerThreadgroup);
  void EndEncoding(void *encoder);
  bool Commit();

#if defined(CUTENN_METAL_DEBUG)
  void MakeCaptureScopeAvailable();
  void BeginCaptureScope();
  void EndCaptureScope();
  void PushCommandEncoderToDebugGroup(void *encoder, const std::string &label);
  void PopCommandEncoderFromDebugGroup(void *encoder);
#endif // CUTENN_METAL_DEBUG

private:
  Context();

  CuteContextImpl *impl_;
};

} // namespace cutenn

#endif // !CUTENN_BASE_CONTEXT_H_
