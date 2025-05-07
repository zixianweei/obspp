#ifndef CUTENN_BASE_TENSOR_H_
#define CUTENN_BASE_TENSOR_H_

#include "base/macros.hpp"
#include "base/types.hpp"

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
  MTLBufferPtr GetRawBuffer();

private:
  TensorImpl *impl_;
};

} // namespace cutenn

#endif // !CUTENN_BASE_TENSOR_H_
