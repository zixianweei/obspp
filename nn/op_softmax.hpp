#ifndef CUTE_NN_SOFTMAX_HPP_
#define CUTE_NN_SOFTMAX_HPP_

#include <memory>

#include "base/platform.hpp"
#include "base/macros.hpp"
#include "nn/op_base.hpp"

namespace cute { namespace nn {

CUTE_CLASS_FORWARD_DECLARATION(SoftmaxImpl);

class OpSoftmax : public OpBase<OpSoftmax>
{
public:
    struct Property
    {
    };

    template <typename... Args>
    OpSoftmax(Args&&... args);

private:
    std::unique_ptr<SoftmaxImpl> impl_;
    std::unique_ptr<Property> property_;
};


}}  // namespace cute::nn

#endif  // !CUTE_NN_SOFTMAX_HPP_
