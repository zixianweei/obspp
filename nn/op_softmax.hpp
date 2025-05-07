#ifndef CUTENN_OP_SOFTMAX_HPP_
#define CUTENN_OP_SOFTMAX_HPP_

#include "nn/op_base.h"

namespace cutenn {

class OpSoftmax : public OpBase {
public:
  OpSoftmax() = default;
  explicit OpSoftmax(int axis) : axis_(axis) {};

  bool Forward(Tensor &src, Tensor &dst) override;
  std::string GetKernelName() const override;

  int GetAxis() const { return axis_; }
  void SetAxis(int axis) { axis_ = axis; }

private:
  int axis_{-1};
};

} // namespace cutenn

#endif // !CUTENN_OP_SOFTMAX_HPP_
