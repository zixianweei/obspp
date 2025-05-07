#ifndef CUTENN_BASE_TYPES_HPP_
#define CUTENN_BASE_TYPES_HPP_

#include <vector>

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

} // namespace cutenn

#endif //! CUTENN_BASE_TYPES_HPP_