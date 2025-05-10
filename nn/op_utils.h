#ifndef CUTENN_NN_OP_UTILS_H_
#define CUTENN_NN_OP_UTILS_H_

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

class Tensor;

#ifdef __OBJC__

// id<MTLBuffer> makeAttribute(const Tensor &src, const Tensor &dst);

#else
#error DO NOT USE THIS FILE IN NON-OBJC BUILDS.
#endif // __OBJC__

#endif // !CUTENN_NN_OP_UTILS_H_