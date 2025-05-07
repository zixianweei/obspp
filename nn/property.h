#ifndef CUTE_PROPERTY_H_
#define CUTE_PROPERTY_H_

namespace cutenn {

struct TensorProperty {
  int i_n;
  int i_c;
  int i_h;
  int i_w;
  int o_n;
  int o_c;
  int o_h;
  int o_w;
};

} // namespace cutenn

#endif // !CUTE_PROPERTY_H_
