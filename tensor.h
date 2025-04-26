#ifndef CUTE_TENSOR_H_
#define CUTE_TENSOR_H_

#include <vector>

#ifdef __OBJC__
@class CuteTensorImpl;
#else
typedef struct objc_object CuteTensorImpl;
#endif

#ifdef __OBJC__
#    include <Foundation/Foundation.h>
#    include <Metal/Metal.h>
typedef id<MTLBuffer> MTLBufferPtr;
#else
typedef void* MTLBufferPtr;
#endif

namespace cute {

enum class Format
{
    kUnknown,
    kUnsignedChar8,
    kFloat32,
};

class Tensor
{
public:
    Tensor();
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    MTLBufferPtr GetRawBuffer();
    bool fromBytes(const void* data, const std::vector<int>& shape, Format format);
    bool toBytes(void** data, const std::vector<int>& shape, Format format);

    std::vector<int> Shape() const;

private:
#ifdef __OBJC__
    __strong CuteTensorImpl* impl_;
#else
    CuteTensorImpl* impl_;
#endif
};

}  // namespace cute

#endif  // !CUTE_TENSOR_H_
