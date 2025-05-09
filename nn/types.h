#ifndef CUTENN_NN_TYPES_H_
#define CUTENN_NN_TYPES_H_

namespace cutenn {

struct OpFlipAttribute {
  int i_h;
  int i_w;
  int o_h;
  int o_w;
};

struct OpSoftmaxAttribute {
  int i_n;
  int i_c;
  int i_h;
  int i_w;
  int o_n;
  int o_c;
  int o_h;
  int o_w;
  int axis;
};

} // namespace cutenn

#endif // !CUTENN_NN_TYPES_H_