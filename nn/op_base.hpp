#ifndef CUTE_NN_OP_BASE_HPP_
#define CUTE_NN_OP_BASE_HPP_

#include "base/macros.hpp"

namespace cute { namespace nn {

CUTE_CLASS_FORWARD_DECLARATION(Tensor);

template <typename OpType>
class OpBase
{
public:
    OpBase() = delete;

    OpBase(const OpBase&) = delete;
    OpBase& operator=(const OpBase&) = delete;
    OpBase(OpBase&&) noexcept = delete;
    OpBase& operator=(OpBase&&) noexcept = delete;


    void Forward(Tensor& src, Tensor& dst);
};

template <typename OpType>
void OpBase<OpType>::Forward(Tensor& src, Tensor& dst)
{
    static_cast<OpType*>(this)->Forward(src, dst);
}

}}  // namespace cute::nn

#endif  // !CUTE_NN_OP_BASE_HPP_
