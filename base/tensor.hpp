#ifndef CUTE_TENSOR_H_
#define CUTE_TENSOR_H_

#include <vector>

#include "base/macros.hpp"

CUTENN_OBJC_FORWARD_DECLARATION(CuteTensorImpl);

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#endif

CUTENN_TYPE_ALIAS(id<MTLBuffer>, MTLBufferPtr);

namespace cute {

enum class Format {
  kUnknown,
  kUnsignedChar8,
  kFloat32,
};

class Tensor {
public:
  Tensor();
  ~Tensor();

  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  MTLBufferPtr GetRawBuffer();
  bool fromBytes(const void *data, const std::vector<int> &shape,
                 Format format);
  bool toBytes(void **data, const std::vector<int> &shape, Format format);

  std::vector<int> Shape() const;

private:
  CuteTensorImpl *impl_;
};

} // namespace cute

#endif // !CUTE_TENSOR_H_
