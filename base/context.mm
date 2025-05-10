#include "context.h"

#include <string>

#include <mach-o/dyld.h>
#include <mach-o/getsect.h>

#include "base/logger.h"
#include "base/types.h"

static dispatch_data_t find_section_data(const std::string &section_name) {
  uint32_t image_idx = 0U;
  uint32_t image_count = _dyld_image_count();
  for (uint32_t i = 0; i < image_count; i++) {
    if (strstr(_dyld_get_image_name(i), "/cutenn")) {
      image_idx = i;
      break;
    }
  }

  const struct mach_header_64 *image_header =
      reinterpret_cast<const struct mach_header_64 *>(
          _dyld_get_image_header(image_idx));

  unsigned long section_size = 0;
  const uint8_t *section_data = getsectiondata(
      image_header, "__TEXT", section_name.c_str(), &section_size);
  if (section_data == nullptr) {
    throw std::runtime_error("Can't find metal library section " +
                             section_name);
  }
  return dispatch_data_create(section_data, section_size,
                              dispatch_get_main_queue(),
                              ^(){
                              });
}

@interface MTL4CuteContext ()
+ (NSString *)toString:(const std::string &)s;
@end

@implementation MTL4CuteContext

- (instancetype)init {
  self = [super init];

  CUTENN_LOG_INFO("{}: allocating", __func__);

#if defined(CUTENN_METAL_DEBUG)
  NSArray *devices = MTLCopyAllDevices();
  for (id<MTLDevice> device in devices) {
    CUTENN_LOG_INFO("{}: found device: {}", __func__,
                    [[device name] UTF8String]);
  }
#endif

  if (_device == nil) {
    _device = MTLCreateSystemDefaultDevice();
    CUTENN_LOG_INFO("{}: using device: {}", __func__,
                    [[_device name] UTF8String]);
  }

  if (_device != nil) {
    _hasSimdGroupReduction = [_device supportsFamily:MTLGPUFamilyApple7];
    // _hasSimdGroupReduction |= [_device supportsFamily:MTLGPUFamilyMetal3];

    _hasBFloat = [_device supportsFamily:MTLGPUFamilyApple6];
    // _hasBFloat |= [_device supportsFamily:MTLGPUFamilyMetal3];
  }

  if (_commandQueue == nil) {
    _commandQueue = [_device newCommandQueue];
  }
  if (_commandQueue == nil) {
    CUTENN_LOG_ERROR("{}: error: failed to create command queue", __func__);
    return nil;
  }

  _queue = dispatch_queue_create("cutenn_metal", DISPATCH_QUEUE_CONCURRENT);

  if (_library == nil) {
    NSError *error = nil;
#if defined(CUTENN_EMBED_METALLIB)
    _library = [_device newLibraryWithData:find_section_data("cutenn_metallib")
                                     error:&error];
#else
    _library =
        [_device newLibraryWithURL:[NSURL fileURLWithPath:@"cutenn.metallib"]
                             error:&error];
#endif // CUTENN_EMBED_METALLIB
    if (error) {
      CUTENN_LOG_ERROR("{}: error: {}", __func__,
                       [[error description] UTF8String]);
      return nil;
    }
    CUTENN_LOG_INFO("{}: create library from section", __func__);
  }

  if (_commandBuffer == nil) {
    _commandBuffer = [_commandQueue commandBuffer];
  }
  if (_commandBuffer == nil) {
    CUTENN_LOG_ERROR("{}: error: failed to create command buffer", __func__);
    return nil;
  }

  _schedCommandBuffer = [[NSMutableArray alloc] init];
  if (_schedCommandBuffer == nil) {
    CUTENN_LOG_ERROR("{}: error: failed to create sched command buffer",
                     __func__);
    return nil;
  }

  return self;
}

- (void)dealloc {
  CUTENN_SAFE_RELEASE(_device);
  CUTENN_SAFE_RELEASE(_commandQueue);
  CUTENN_SAFE_RELEASE(_library);
  CUTENN_SAFE_RELEASE(_queue);
  CUTENN_SAFE_RELEASE(_cachedCPS);
  CUTENN_SAFE_RELEASE(_commandBuffer);
  CUTENN_SAFE_RELEASE(_schedCommandBuffer);
#if defined(CUTENN_METAL_DEBUG)
  CUTENN_SAFE_RELEASE(_captureScope);
#endif // CUTENN_METAL_DEBUG
  [super dealloc];
}

- (id<MTLComputePipelineState>)findComputePipelineState:(NSString *)kernelName {
  // INFO: kernelName is kernel_***
  if (kernelName == nil) {
    CUTENN_LOG_ERROR("{}: kernel name is nil", __func__);
    return nil;
  }

  id<MTLComputePipelineState> cps = _cachedCPS[kernelName];
  if (cps != nil) {
    CUTENN_LOG_INFO("{}: find cached compute pipeline state: {}", __func__,
                    [kernelName UTF8String]);
    return cps;
  }

  id<MTLFunction> function = [_library newFunctionWithName:kernelName];
  if (function == nil) {
    CUTENN_LOG_ERROR("{}: failed to create function", __func__);
    return nil;
  }

  NSError *error = nil;
  cps = [_device newComputePipelineStateWithFunction:function error:&error];
  if (error != nil) {
    CUTENN_LOG_ERROR("{}: error: {}", __func__,
                     [[error description] UTF8String]);
    return nil;
  }
  _cachedCPS[kernelName] = cps;
  return cps;
}

- (id<MTLComputeCommandEncoder>)createEncoder {
  if (_commandBuffer == nil) {
    CUTENN_LOG_ERROR("{}: command buffer is nil", __func__);
    return nil;
  }
  [_commandBuffer enqueue];
  return [_commandBuffer computeCommandEncoder];
}

- (BOOL)commit {
  [_commandBuffer commit];
  [_commandBuffer waitUntilCompleted];
  _commandBuffer = [_commandQueue commandBuffer];
  return TRUE;
}

+ (NSString *)toString:(const std::string &)s {
  return [NSString stringWithCString:s.c_str()
                            encoding:[NSString defaultCStringEncoding]];
}

#if defined(CUTENN_METAL_DEBUG)
- (void)makeCaptureScopeAvailable {
  if ([self isCaptureScopeOn]) {
    return;
  }
  [self setCaptureScope:[[MTLCaptureManager sharedCaptureManager]
                            newCaptureScopeWithDevice:[self device]]];
  if ([self captureScope] == nil) {
    CUTENN_LOG_ERROR("{}: failed to create capture scope", __func__);
    return;
  }

  MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
  descriptor.captureObject = [self captureScope];
  descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
  descriptor.outputURL =
      [NSURL fileURLWithPath:[NSString stringWithFormat:@"cutenn.gputrace"]];

  NSError *error = nil;
  if (![[MTLCaptureManager sharedCaptureManager]
          startCaptureWithDescriptor:descriptor
                               error:&error]) {
    CUTENN_LOG_ERROR("{}: failed to start capture: {}", __func__,
                     [[error description] UTF8String]);
  }
  [self setIsCaptureScopeOn:YES];
}

- (void)beginCapture {
  if ([self isCaptureScopeOn]) {
    return;
  }
  [[self captureScope] beginScope];
}

- (void)endCapture {
  if (![self isCaptureScopeOn]) {
    return;
  }
  [[self captureScope] endScope];
  [[MTLCaptureManager sharedCaptureManager] stopCapture];
}

#endif // CUTENN_METAL_DEBUG

@end

namespace cutenn {

// static
ContextOwner &ContextOwner::GetInstance() {
  static ContextOwner instance;
  return instance;
}

ContextOwner::ContextOwner() {
  context_ = [[MTL4CuteContext alloc] init];
  if (context_ == nil) {
    CUTENN_LOG_ERROR("{}: failed to create metal context", __func__);
    return;
  }
}

ContextOwner::~ContextOwner() { CUTENN_SAFE_RELEASE(context_); }

} // namespace cutenn
