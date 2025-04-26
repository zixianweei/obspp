#ifndef CUTE_OP_BASE_H_
#define CUTE_OP_BASE_H_

#include <string>

namespace cute {

class Tensor;

class OpBase
{
public:
    virtual bool Forward(Tensor& src, Tensor& dst) = 0;
    virtual std::string GetKernelName() const = 0;
};

}  // namespace cute

#endif  // !CUTE_OP_BASE_H_
