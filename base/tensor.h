#ifndef CUTENN_BASE_TENSOR_H_
#define CUTENN_BASE_TENSOR_H_

#include "base/macros.h"
#include "base/types.h"

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#endif // __OBJC__

CUTENN_OBJC_FORWARD_DECLARATION(TensorImpl);

namespace cutenn {

class Tensor {
public:
  Tensor(Format format = Format::kFloat32);
  Tensor(const void *data, const TShape &shape,
         Format format = Format::kFloat32);
  ~Tensor();

  Tensor(const Tensor &rhs);
  Tensor &operator=(const Tensor &rhs);
  Tensor(Tensor &&rhs) noexcept;
  Tensor &operator=(Tensor &&rhs) noexcept;

  bool Upload(const void *data, const TShape &shape,
              Format format = Format::kFloat32);
  bool Download(void *data, const TShape &shape,
                Format format = Format::kFloat32);

  TShape GetShape() const;
  size_t GetDims() const;

#ifdef __OBJC__
  id<MTLBuffer> GetRawBuffer();
#endif // __OBJC__

private:
  TensorImpl *impl_;
};

} // namespace cutenn

#endif // !CUTENN_BASE_TENSOR_H_
