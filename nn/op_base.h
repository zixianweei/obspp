#ifndef CUTE_OP_BASE_H_
#define CUTE_OP_BASE_H_

#include <string>

#include "base/macros.hpp"
#include "base/tensor.hpp"

namespace cute {

class Tensor;

class OpBase {
public:
  virtual bool Forward(Tensor &src, Tensor &dst) = 0;
  virtual std::string GetKernelName() const = 0;
  MTLBufferPtr AllocateTensorProperty(const Tensor &src, const Tensor &dst);
};

} // namespace cute

#endif // !CUTE_OP_BASE_H_
