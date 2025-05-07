#include "op_base.h"

#include "base/context.hpp"
#include "base/logger.hpp"
#include "property.h"

namespace cutenn {

MTLBufferPtr OpBase::AllocateTensorProperty(const Tensor &src,
                                            const Tensor &dst) {
  TensorProperty property;
  property.i_n = src.GetShape()[0];
  property.i_c = src.GetShape()[1];
  property.i_h = src.GetShape()[2];
  property.i_w = src.GetShape()[3];
  property.o_n = dst.GetShape()[0];
  property.o_c = dst.GetShape()[1];
  property.o_h = dst.GetShape()[2];
  property.o_w = dst.GetShape()[3];

  return Context::GetInstance().MakeBuffer((const void *)&property,
                                           sizeof(TensorProperty));
}

} // namespace cutenn
