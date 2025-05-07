#include "nn/op_softmax.hpp"

#include "base/logger.hpp"

namespace cutenn {

bool OpSoftmax::Forward(Tensor &src, Tensor &dst) {
  // TODO: implement
  return true;
}

std::string OpSoftmax::GetKernelName() const {
  int axis = GetAxis();
  switch (axis) {
  case 0:
    return "kernel_softmax_axis_0";
  case 1:
    return "kernel_softmax_axis_1";
  case 2:
    return "kernel_softmax_axis_2";
  case 3:
    return "kernel_softmax_axis_3";
  default:
    break;
  }
  CUTENN_LOG_CRITICAL("{}: axis {} is not supported", __func__, axis);
  return "";
}

} // namespace cutenn
