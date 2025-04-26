#include "context.h"

#include <string>

#include "logger.h"
#include "macros.h"

#pragma mark - ContextImpl

typedef NSMutableDictionary<NSString*, id<MTLComputePipelineState>>*
    ComputePipelineStateDictionary;
typedef NSMutableArray<id<MTLCommandBuffer>>* CommandBufferArray;

@interface CuteContextImpl : NSObject
@property (strong, nonatomic) id<MTLDevice> device;
@property (strong, nonatomic) id<MTLCommandQueue> commandQueue;
@property (strong, nonatomic) id<MTLLibrary> library;
@property (assign, nonatomic) BOOL hasSimdGroupReduction;
@property (assign, nonatomic) BOOL hasBFloat;
@property (strong, nonatomic) dispatch_queue_t queue;
@property (strong, nonatomic) ComputePipelineStateDictionary cachedCPS;
@property (strong, nonatomic) id<MTLCommandBuffer> commandBuffer;
@property (strong, nonatomic) CommandBufferArray schedCommandBuffer;
@property (strong, nonatomic) id<MTLCaptureScope> captureScope;

#if !__has_feature(objc_arc)
- (void)dealloc;
#endif
- (id<MTLComputePipelineState>)findComputePipelineState:(NSString*)kernelName;
- (id<MTLComputeCommandEncoder>)createEncoder;
- (BOOL)commit;
@end

@interface CuteContextImpl ()
+ (NSString*)toString:(const std::string&)s;
@end

@implementation CuteContextImpl

- (nonnull instancetype)init
{
    self = [super init];

    CUTE_LOG_INFO("{}: allocating", __func__);

#if defined(CUTE_DEBUG)
    NSArray* devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        CUTE_LOG_INFO("{}: found device: {}", __func__, [[device name] UTF8String]);
    }
#endif

    if (_device == nil) {
        _device = MTLCreateSystemDefaultDevice();
        CUTE_LOG_INFO("{}: using device: {}", __func__,
            [[_device name] UTF8String]);
    }

    _captureScope = [[MTLCaptureManager sharedCaptureManager] defaultCaptureScope];

    if (_device != nil) {
        _hasSimdGroupReduction = [_device supportsFamily:MTLGPUFamilyApple7];
        _hasSimdGroupReduction |= [_device supportsFamily:MTLGPUFamilyMetal3];

        _hasBFloat = [_device supportsFamily:MTLGPUFamilyApple6];
        _hasBFloat |= [_device supportsFamily:MTLGPUFamilyMetal3];
    }

    if (_commandQueue == nil) {
        _commandQueue = [_device newCommandQueue];
    }
    if (_commandQueue == nil) {
        CUTE_LOG_ERROR("{}: error: failed to create command queue", __func__);
        return nil;
    }

    _queue = dispatch_queue_create("OpenCV_Metal", DISPATCH_QUEUE_CONCURRENT);

    if (_library == nil) {
        @autoreleasepool {
            // if not found the metal library
            // then load then from metal files.
            NSError* error;
            NSString* libraryPath =
                [NSString stringWithFormat:@"%@/source/nnn/cute/flip.metallib",
                    NSHomeDirectory()];
            if (libraryPath != nil) {
                _library =
                    [_device newLibraryWithURL:[NSURL fileURLWithPath:libraryPath]
                                         error:&error];
                if (error) {
                    CUTE_LOG_ERROR("{}: error: {}", __func__,
                        [[error description] UTF8String]);
                    return nil;
                }
            } else {
                CUTE_LOG_INFO("{}: cute.metallib not found, loading from source",
                    __func__);
                NSString* sourcePath;
            }
        }
    }

    if (_commandBuffer == nil) {
        _commandBuffer = [_commandQueue commandBuffer];
    }
    if (_commandBuffer == nil) {
        CUTE_LOG_ERROR("{}: error: failed to create command buffer", __func__);
        return nil;
    }

    _schedCommandBuffer = [[NSMutableArray alloc] init];
    if (_schedCommandBuffer == nil) {
        CUTE_LOG_ERROR("{}: error: failed to create sched command buffer",
            __func__);
        return nil;
    }

    return self;
}

#if !__has_feature(objc_arc)
- (void)dealloc
{
    SAFE_RELEASE(_device);
    SAFE_RELEASE(_commandQueue);
    SAFE_RELEASE(_library);
    SAFE_RELEASE(_queue);
    SAFE_RELEASE(_cachedCPS);
    SAFE_RELEASE(_commandBuffer);
    SAFE_RELEASE(_schedCommandBuffer);
    SAFE_RELEASE(_captureScope);
    [super dealloc];
}
#endif

- (id<MTLComputePipelineState>)findComputePipelineState:(NSString*)kernelName
{
    // INFO: kernelName is kernel_***
    if (kernelName == nil) {
        CUTE_LOG_ERROR("{}: kernel name is nil", __func__);
        return nil;
    }

    id<MTLComputePipelineState> cps = _cachedCPS[kernelName];
    if (cps != nil) {
        CUTE_LOG_INFO("{}: find cached compute pipeline state: {}", __func__,
            [kernelName UTF8String]);
        return cps;
    }

    id<MTLFunction> function = [_library newFunctionWithName:kernelName];
    if (function == nil) {
        CUTE_LOG_ERROR("{}: failed to create function", __func__);
        return nil;
    }

    NSError* error = nil;
    cps = [_device newComputePipelineStateWithFunction:function error:&error];
    if (error != nil) {
        CUTE_LOG_ERROR("{}: error: {}", __func__, [[error description] UTF8String]);
        return nil;
    }
    _cachedCPS[kernelName] = cps;
    return cps;
}

- (id<MTLComputeCommandEncoder>)createEncoder
{
    if (_commandBuffer == nil) {
        CUTE_LOG_ERROR("{}: command buffer is nil", __func__);
        return nil;
    }
    [_commandBuffer enqueue];
    return [_commandBuffer computeCommandEncoder];
}

- (BOOL)commit
{
    [_commandBuffer commit];
    [_commandBuffer waitUntilCompleted];
    _commandBuffer = [_commandQueue commandBuffer];
    return TRUE;
}

+ (NSString*)toString:(const std::string&)s
{
    return [NSString stringWithCString:s.c_str()
                              encoding:[NSString defaultCStringEncoding]];
}

@end

#pragma mark - Context

namespace cute {

// static
Context& Context::GetInstance()
{
    static Context instance;
    return instance;
}

Context::Context()
{
    impl_ = [[CuteContextImpl alloc] init];
}

Context::~Context()
{
#if !__has_feature(objc_arc)
    [impl_ release];
#endif
    impl_ = nullptr;
}

MTLDevicePtr Context::GetDevice()
{
    return [impl_ device];
}

MTLCommandQueuePtr Context::GetCommandQueue()
{
    return [impl_ commandQueue];
}

MTLComputePipelineStatePtr Context::findComputePipelineState(
    const std::string& kname)
{
    return [impl_ findComputePipelineState:[CuteContextImpl toString:kname]];
}

MTLComputeCommandEncoderPtr Context::GetCommandEncoder()
{
    return [impl_ createEncoder];
}

bool Context::Commit()
{
    return [impl_ commit];
}

void Context::BeginCapture()
{
    [[impl_ captureScope] beginScope];
}

void Context::EndCapture()
{
    [[impl_ captureScope] endScope];
}

MTLSize Context::CalculateGridSize(const std::vector<int>& shape)
{
    int width = shape.size() >= 1 ? shape.back() : 1;
    int height = shape.size() >= 2 ? shape[shape.size() - 2] : 1;
    int depth = 1;
    for (int i = 0; i < int(shape.size()) - 2; ++i) {
        depth *= shape[i];
    }
    return MTLSizeMake(width, height, depth);
}

MTLSize Context::CalculateThreadgroupSize(const MTLSize& gridSize)
{
    MTLSize maxSize = [[impl_ device] maxThreadsPerThreadgroup];
    MTLSize groupSize = MTLSizeMake(1, 1, 1);

    // 优先填充 width 维度
    groupSize.width = MIN(maxSize.width, gridSize.width);

    // 剩余容量分配给 height
    groupSize.height = MIN(maxSize.height, gridSize.height);
    if (groupSize.width * groupSize.height > maxSize.width) {
        groupSize.height = 1;
    }

    // 最后填充 depth
    groupSize.depth = MIN(maxSize.depth, gridSize.depth);
    if (groupSize.width * groupSize.height * groupSize.depth > maxSize.width) {
        groupSize.depth = 1;
    }

    return groupSize;
}

} // namespace cute
