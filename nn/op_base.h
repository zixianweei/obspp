#ifndef CUTENN_NN_OP_BASE_H_
#define CUTENN_NN_OP_BASE_H_

#include "base/types.hpp"

namespace cutenn {

class Tensor;

class OpBase {
public:
  virtual bool Forward(Tensor &src, Tensor &dst) = 0;
  virtual MTLBufferPtr MakeAttribute(Tensor &src, Tensor &dst) = 0;
};

} // namespace cutenn

#endif // !CUTENN_NN_OP_BASE_H_
