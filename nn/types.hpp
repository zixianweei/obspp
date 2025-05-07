#ifndef CUTENN_NN_TYPES_HPP_
#define CUTENN_NN_TYPES_HPP_

namespace cutenn {

struct OpFlipAttribute {
  int i_h;
  int i_w;
  int o_h;
  int o_w;
};

struct OpSoftmaxAttribute {
  int axis;
};

} // namespace cutenn

#endif // !CUTENN_NN_TYPES_HPP_