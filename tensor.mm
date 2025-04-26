#include "tensor.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#include <vector>

#include "context.h"
#include "logger.h"

#if !__has_feature(objc_arc)
#error "ARC is off"
#endif

@interface CuteTensorImpl : NSObject
@property (strong, nonatomic) id<MTLBuffer> buffer;
@property (assign, nonatomic) cute::Format format;
@property (assign, nonatomic) std::vector<int> shape;

- (BOOL)fromBytes:(const void*)data
            shape:(const std::vector<int>&)shape
           format:(cute::Format)format;
- (BOOL)toBytes:(void**)data
          shape:(const std::vector<int>&)shape
         format:(cute::Format)format;

+ (int)shapeCount:(const std::vector<int>&)shape;
+ (int)elementSize:(cute::Format)format;
@end

@implementation CuteTensorImpl

- (nonnull instancetype)init
{
    self = [super init];
    if (self != nil) {
        _buffer = nil;
        _format = cute::Format::kUnknown;
        _shape = {};
    }
    return self;
}

- (BOOL)fromBytes:(const void*)data
            shape:(const std::vector<int>&)shape
           format:(cute::Format)format
{
    id<MTLDevice> device = cute::Context::GetInstance().GetDevice();
    if (device == nil) {
        CUTE_LOG_ERROR("{}: context device is nil", __func__);
        return FALSE;
    }

    _shape = shape;
    _format = format;

    int len =
        [CuteTensorImpl shapeCount:_shape] * [CuteTensorImpl elementSize:_format];
    if (len <= 0) {
        CUTE_LOG_ERROR("{}: invalid buffer length", __func__);
        return FALSE;
    }

    _buffer = [device newBufferWithLength:len
                                  options:MTLResourceStorageModeShared];
    if (_buffer == nil) {
        CUTE_LOG_ERROR("{}: buffer is nil", __func__);
        return FALSE;
    }

    if (data != nullptr) {
        std::ignore = memcpy(_buffer.contents, data, len);
    }

    return TRUE;
}

- (BOOL)toBytes:(void**)data
          shape:(const std::vector<int>&)shape
         format:(cute::Format)format
{
    int targetBufferLen =
        [CuteTensorImpl shapeCount:shape] * [CuteTensorImpl elementSize:format];
    int sourceBufferLen =
        [CuteTensorImpl shapeCount:_shape] * [CuteTensorImpl elementSize:_format];
    if (targetBufferLen != sourceBufferLen) {
        CUTE_LOG_ERROR("{}: buffer of source({}) and target({}) is not same",
            __func__, sourceBufferLen, targetBufferLen);
        return FALSE;
    }

    std::ignore = memcpy(*data, _buffer.contents, targetBufferLen);

    return TRUE;
}

+ (int)shapeCount:(const std::vector<int>&)shape
{
    int count = 1;
    for (const int& e : shape) {
        count *= e;
    }
    return count;
}

+ (int)elementSize:(cute::Format)format
{
    switch (format) {
    case cute::Format::kUnsignedChar8:
        return 1;
    case cute::Format::kFloat32:
        return 4;
    default:
        break;
    }
    CUTE_LOG_CRITICAL("{}: elementSize unreachable", __func__);
    return 0;
}

@end

namespace cute {

Tensor::Tensor()
{
    impl_ = [[CuteTensorImpl alloc] init];
}

Tensor::~Tensor()
{
    impl_ = nullptr;
}

MTLBufferPtr Tensor::GetRawBuffer()
{
    return [impl_ buffer];
}

bool Tensor::fromBytes(const void* data,
    const std::vector<int>& shape,
    Format format)
{
    return [impl_ fromBytes:data shape:shape format:format];
}

bool Tensor::toBytes(void** data,
    const std::vector<int>& shape,
    Format format)
{
    return [impl_ toBytes:data shape:shape format:format];
}

std::vector<int> Tensor::Shape() const
{
    return [impl_ shape];
}

} // namespace cute
