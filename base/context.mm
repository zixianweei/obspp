#include "context.hpp"

#include <string>

#include <mach-o/dyld.h>
#include <mach-o/getsect.h>

#include "base/logger.hpp"
#include "base/types.hpp"

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

inline MTLSize size_to_mtl_size(const cutenn::Size &size) {
  return MTLSizeMake(size.x, size.y, size.z);
}

typedef NSMutableDictionary<NSString *, id<MTLComputePipelineState>>
    *ComputePipelineStateDictionary;
typedef NSMutableArray<id<MTLCommandBuffer>> *CommandBufferArray;

@interface CuteContextImpl : NSObject
@property(strong, nonatomic) id<MTLDevice> device;
@property(strong, nonatomic) id<MTLCommandQueue> commandQueue;
@property(strong, nonatomic) id<MTLLibrary> library;
@property(assign, nonatomic) BOOL hasSimdGroupReduction;
@property(assign, nonatomic) BOOL hasBFloat;
@property(strong, nonatomic) dispatch_queue_t queue;
@property(strong, nonatomic) ComputePipelineStateDictionary cachedCPS;
@property(strong, nonatomic) id<MTLCommandBuffer> commandBuffer;
@property(strong, nonatomic) CommandBufferArray schedCommandBuffer;
#if defined(CUTENN_METAL_DEBUG)
@property(strong, nonatomic) id<MTLCaptureScope> captureScope;
@property(assign, nonatomic) BOOL isCaptureScopeOn;
#endif // CUTENN_METAL_DEBUG

- (void)dealloc;
- (id<MTLComputePipelineState>)findComputePipelineState:(NSString *)kernelName;
- (id<MTLComputeCommandEncoder>)createEncoder;
- (BOOL)commit;
#if defined(CUTENN_METAL_DEBUG)
- (void)SwitchOnCapture;
- (void)SwitchOffCapture;
#endif // CUTENN_METAL_DEBUG
@end

@interface CuteContextImpl ()
+ (NSString *)toString:(const std::string &)s;
@end

@implementation CuteContextImpl

- (nonnull instancetype)init {
  self = [super init];

  CUTENN_LOG_INFO("{}: allocating", __func__);

#if defined(CUTE_DEBUG)
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
    // _library = [_device newLibraryWithData:find_section_data("metal_basic")
    //                                  error:&error];
    _library = [_device newLibraryWithFile:@"cutenn.metallib" error:&error];
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
- (void)SwitchOnCapture {
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
  // descriptor.outputURL = [NSURL fileURLWithPath:[NSString
  // stringWithFormat:@"/tmp/%@.gputrace", [NSUUID UUID].UUIDString]];
  descriptor.outputURL = [NSURL
      fileURLWithPath:[NSString stringWithFormat:@"/tmp/cutenn.gputrace"]];

  NSError *error = nil;
  if (![[MTLCaptureManager sharedCaptureManager]
          startCaptureWithDescriptor:descriptor
                               error:&error]) {
    CUTENN_LOG_ERROR("{}: failed to start capture: {}", __func__,
                     [[error description] UTF8String]);
  }
  [self setIsCaptureScopeOn:YES];
  [[self captureScope] beginScope];
}

- (void)SwitchOffCapture {
  if (![self isCaptureScopeOn]) {
    return;
  }
  [[self captureScope] endScope];
  [[MTLCaptureManager sharedCaptureManager] stopCapture];
  [self setIsCaptureScopeOn:NO];
}

#endif // CUTENN_METAL_DEBUG

@end

namespace cutenn {

// static
Context &Context::GetInstance() {
  static Context instance;
  return instance;
}

Context::Context() { impl_ = [[CuteContextImpl alloc] init]; }

Context::~Context() {
#if !__has_feature(objc_arc)
  [impl_ release];
#endif
  impl_ = nullptr;
}

MTLDevicePtr Context::GetDevice() { return [impl_ device]; }

MTLCommandQueuePtr Context::GetCommandQueue() { return [impl_ commandQueue]; }

MTLComputePipelineStatePtr
Context::GetComputePipelineState(const std::string &kname) {
  return [impl_ findComputePipelineState:[CuteContextImpl toString:kname]];
}

MTLComputeCommandEncoderPtr Context::GetCommandEncoder() {
  return [impl_ createEncoder];
}

MTLBufferPtr Context::MakeBuffer(const void *data, size_t size) {
  MTLBufferPtr buffer =
      [[impl_ device] newBufferWithBytes:data
                                  length:size
                                 options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    CUTENN_LOG_ERROR("{}: failed to allocate property buffer.", __func__);
    return nil;
  }
  return buffer;
}

void Context::SetCommandEncoderComputePipelineState(void *encoder,
                                                    void *state) {
  [static_cast<MTLComputeCommandEncoderPtr>(encoder)
      setComputePipelineState:static_cast<MTLComputePipelineStatePtr>(state)];
}

void Context::SetCommandEncoderBuffer(void *encoder, void *buffer, int offset,
                                      int index) {
  [static_cast<MTLComputeCommandEncoderPtr>(encoder)
      setBuffer:static_cast<MTLBufferPtr>(buffer)
         offset:offset
        atIndex:index];
}

unsigned int Context::GetMaxTotalThreadsPerThreadgroup(void *state) {
  return [static_cast<MTLComputePipelineStatePtr>(state)
      maxTotalThreadsPerThreadgroup];
}

unsigned int Context::GetThreadExecutionWidth(void *state) {
  return [static_cast<MTLComputePipelineStatePtr>(state) threadExecutionWidth];
}

void Context::DispatchThreads(void *encoder, const Size &threads,
                              const Size &threadsPerThreadgroup) {
  [static_cast<MTLComputeCommandEncoderPtr>(encoder)
            dispatchThreads:size_to_mtl_size(threads)
      threadsPerThreadgroup:size_to_mtl_size(threadsPerThreadgroup)];
}

void Context::EndEncoding(void *encoder) {
  [static_cast<MTLComputeCommandEncoderPtr>(encoder) endEncoding];
}

bool Context::Commit() { return [impl_ commit]; }

#if defined(CUTENN_METAL_DEBUG)

void Context::MakeCaptureScopeAvailable() { [impl_ SwitchOnCapture]; }

void Context::BeginCaptureScope() {
  if (![impl_ isCaptureScopeOn]) {
    CUTENN_LOG_INFO("{}: capture scope is not available", __func__);
    return;
  }
  [[impl_ captureScope] beginScope];
}

void Context::EndCaptureScope() {
  if (![impl_ isCaptureScopeOn]) {
    CUTENN_LOG_INFO("{}: capture scope is not available", __func__);
    return;
  }
  [[impl_ captureScope] endScope];
}

void Context::PushCommandEncoderToDebugGroup(void *encoder,
                                             const std::string &label) {
  [static_cast<MTLComputeCommandEncoderPtr>(encoder)
      pushDebugGroup:[NSString stringWithCString:label.c_str()
                                        encoding:NSUTF8StringEncoding]];
}

void Context::PopCommandEncoderFromDebugGroup(void *encoder) {
  [static_cast<MTLComputeCommandEncoderPtr>(encoder) popDebugGroup];
}

#endif // CUTENN_METAL_DEBUG

} // namespace cutenn
