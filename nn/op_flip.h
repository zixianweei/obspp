#ifndef CUTENN_NN_OP_FLIP_H_
#define CUTENN_NN_OP_FLIP_H_

#include "op_base.h"

namespace cutenn {

class OpFlip : public OpBase {
public:
  bool Forward(Tensor &src, Tensor &dst) override;
  std::string GetKernelName() const override;
};

} // namespace cutenn

#endif // !CUTENN_NN_OP_FLIP_H_
