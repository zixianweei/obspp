#include "op_base.h"

#include "base/logger.h"
#include "context.h"
#include "property.h"

namespace cute {

MTLBufferPtr OpBase::AllocateTensorProperty(const Tensor &src,
                                            const Tensor &dst) {
  TensorProperty property;
  property.i_n = src.Shape()[0];
  property.i_c = src.Shape()[1];
  property.i_h = src.Shape()[2];
  property.i_w = src.Shape()[3];
  property.o_n = dst.Shape()[0];
  property.o_c = dst.Shape()[1];
  property.o_h = dst.Shape()[2];
  property.o_w = dst.Shape()[3];

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

} // namespace cute
