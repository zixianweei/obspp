#ifndef CUTENN_NN_OP_FLIP_H_
#define CUTENN_NN_OP_FLIP_H_

#include <string>

#include "op_base.h"

namespace cutenn {

class OpFlip final : public OpBase {
public:
  bool Forward(Tensor &src, Tensor &dst) override;
  std::string GetKernelName() const;
};

} // namespace cutenn

#endif // !CUTENN_NN_OP_FLIP_H_
