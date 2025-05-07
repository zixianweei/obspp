#ifndef CUTENN_BASE_TYPES_HPP_
#define CUTENN_BASE_TYPES_HPP_

#include <vector>

#include "base/macros.hpp"

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#endif

CUTENN_TYPE_ALIAS(id<MTLDevice>, MTLDevicePtr);
CUTENN_TYPE_ALIAS(id<MTLCommandQueue>, MTLCommandQueuePtr);
CUTENN_TYPE_ALIAS(id<MTLComputePipelineState>, MTLComputePipelineStatePtr);
CUTENN_TYPE_ALIAS(id<MTLComputeCommandEncoder>, MTLComputeCommandEncoderPtr);
CUTENN_TYPE_ALIAS(id<MTLBuffer>, MTLBufferPtr);

namespace cutenn {

enum class Platform {
  kUnknown = 0,
  kCpuOnly,
  kMetal,
};

enum class Format {
  kUnknown = 0,
  kUnsignedChar8,
  kFloat32,
  kFloat16,
};

using TShape = std::vector<int>;

struct Size {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

inline Size MakeSize(unsigned int x, unsigned int y, unsigned int z) {
  return {.x = x, .y = y, .z = z};
}

} // namespace cutenn

#endif //! CUTENN_BASE_TYPES_HPP_