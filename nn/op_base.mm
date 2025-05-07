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

  MTLBufferPtr property_buffer = [Context::GetInstance().GetDevice()
      newBufferWithBytes:(const void *)(&property)
                  length:sizeof(TensorProperty)
                 options:MTLResourceCPUCacheModeWriteCombined];
  if (property_buffer == nil) {
    CUTENN_LOG_ERROR("{}: failed to allocate property buffer.", __func__);
    return nil;
  }
  return property_buffer;
}

} // namespace cutenn
