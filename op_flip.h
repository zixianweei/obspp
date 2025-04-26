#ifndef CUTE_OP_FLIP_H_
#define CUTE_OP_FLIP_H_

#include "op_base.h"

namespace cute {

class OpFlip : public OpBase
{
public:
    bool Forward(Tensor& src, Tensor& dst) override;
    std::string GetKernelName() const override;
};

}  // namespace cute

#endif  // !CUTE_OP_FLIP_H_
