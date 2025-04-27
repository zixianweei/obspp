#ifndef CUTE_NN_SOFTMAX_IMPL_HPP_
#define CUTE_NN_SOFTMAX_IMPL_HPP_

#include "base/platform.hpp"

namespace cute { namespace nn {

class SoftmaxImpl
{
public:
    SoftmaxImpl(base::Platform platform);

private:
    template <base::Platform platform>
    void Create();

    template <>
    void Create<base::Platform::kCpuOnly>();
};

template <>
inline void SoftmaxImpl::Create<base::Platform::kCpuOnly>()
{
    return;
}

}}  // namespace cute::nn

#endif  // !CUTE_NN_SOFTMAX_IMPL_HPP_
