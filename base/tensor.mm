#include "tensor.hpp"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#include <vector>

#include "base/context.hpp"
#include "base/logger.hpp"

namespace {

int shape_count(const cutenn::TShape &shape) {
  int count = 1;
  for (const int &e : shape) {
    count *= e;
  }
  return count;
}

int element_size(cutenn::Format format) {
  switch (format) {
  case cutenn::Format::kUnsignedChar8:
    return 1;
  case cutenn::Format::kFloat32:
    return 4;
  default:
    break;
  }
  CUTENN_LOG_CRITICAL("{}: element_size unreachable", __func__);
  return 0;
}

} // namespace

@interface TensorImpl : NSObject
@property(strong, nonatomic) id<MTLBuffer> buffer;
@property(assign, nonatomic) cutenn::Format format;
@property(assign, nonatomic) cutenn::TShape shape;
@property(assign, nonatomic) int sizeInBytes;

- (instancetype)init;
- (instancetype)initWithFormat:(cutenn::Format)format;
- (void)dealloc;
- (BOOL)upload:(const void *)data
         shape:(const cutenn::TShape &)shape
        format:(cutenn::Format)format;
- (BOOL)download:(void *)data
           shape:(const cutenn::TShape &)shape
          format:(cutenn::Format)format;
@end

@implementation TensorImpl

- (nonnull instancetype)init {
  self = [super init];
  if (self != nil) {
    _buffer = nil;
    _format = cutenn::Format::kUnknown;
    _shape = {};
  }
  return self;
}

- (instancetype)initWithFormat:(cutenn::Format)format {
  self = [super init];
  if (self) {
    _buffer = nil;
    _shape = {};
    _format = format;
  }
  return self;
}

- (void)dealloc {
  CUTENN_SAFE_RELEASE(_buffer);
  [super dealloc];
}

- (BOOL)upload:(const void *)data
         shape:(const cutenn::TShape &)shape
        format:(cutenn::Format)format {
  id<MTLDevice> device = cutenn::Context::GetInstance().GetDevice();
  if (device == nil) {
    CUTENN_LOG_ERROR("{}: context device is nil", __func__);
    return FALSE;
  }

  _shape = shape;
  _format = format;

  int dataLen = shape_count(shape) * element_size(format);
  if (dataLen <= 0) {
    CUTENN_LOG_ERROR("{}: invalid buffer length", __func__);
    return FALSE;
  }

  if (dataLen > [self sizeInBytes]) {
    id<MTLBuffer> buffer = [self buffer];
    CUTENN_SAFE_RELEASE(buffer);
    [self setBuffer:[device newBufferWithLength:dataLen
                                        options:MTLResourceStorageModeShared]];
    if ([self buffer] == nil) {
      CUTENN_LOG_ERROR("{}: buffer is nil", __func__);
      return FALSE;
    }
  }

  [self setSizeInBytes:dataLen];

  if (data != nullptr) {
    std::ignore = memcpy(_buffer.contents, data, dataLen);
  }

  return TRUE;
}

- (BOOL)download:(void *)data
           shape:(const cutenn::TShape &)shape
          format:(cutenn::Format)format {
  int dataLen = shape_count(shape) * element_size(format);
  if (dataLen > [self sizeInBytes]) {
    CUTENN_LOG_ERROR("{}: data is larger than buffer. data length = {}",
                     __func__, dataLen);
    return FALSE;
  }

  std::ignore = memcpy(data, [self buffer].contents, dataLen);
  return TRUE;
}

@end

namespace cutenn {

Tensor::Tensor(Format format) {
  impl_ = [[TensorImpl alloc] initWithFormat:format];
}

Tensor::Tensor(const void *data, const TShape &shape, Format format) {
  impl_ = [[TensorImpl alloc] init];
  [impl_ upload:data shape:shape format:format];
}

Tensor::~Tensor() { CUTENN_SAFE_RELEASE(impl_); }

Tensor::Tensor(const Tensor &rhs) {
  impl_ = [[TensorImpl alloc] init];
  [impl_ upload:[rhs.impl_ buffer].contents
          shape:[rhs.impl_ shape]
         format:[rhs.impl_ format]];
}

Tensor &Tensor::operator=(const Tensor &rhs) {
  if (this != &rhs) {
    CUTENN_SAFE_RELEASE(impl_);
    impl_ = [[TensorImpl alloc] init];
    [impl_ upload:[rhs.impl_ buffer].contents
            shape:[rhs.impl_ shape]
           format:[rhs.impl_ format]];
  }
  return *this;
}

Tensor::Tensor(Tensor &&rhs) noexcept { std::swap(impl_, rhs.impl_); }

Tensor &Tensor::operator=(Tensor &&rhs) noexcept {
  if (this != &rhs) {
    CUTENN_SAFE_RELEASE(impl_);
    std::swap(impl_, rhs.impl_);
  }
  return *this;
}

bool Tensor::Upload(const void *data, const TShape &shape, Format format) {
  return [impl_ upload:data shape:shape format:format];
}

bool Tensor::Download(void *data, const TShape &shape, Format format) {
  return [impl_ download:data shape:shape format:format];
}

TShape Tensor::GetShape() const { return [impl_ shape]; }

size_t Tensor::GetDims() const { return [impl_ shape].size(); }

MTLBufferPtr Tensor::GetRawBuffer() { return [impl_ buffer]; }

} // namespace cutenn
