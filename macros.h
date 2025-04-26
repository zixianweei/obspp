#ifndef CUTE_MACROS_H_
#define CUTE_MACROS_H_

#define SAFE_RELEASE(__object__) \
    if (__object__ != nil) {     \
        [__object__ release];    \
    }

#ifdef __OBJC__
#define CUTE_OBJC_FORWARD_DECLARATION(__class__) @class __class__
#else
#define CUTE_OBJC_FORWARD_DECLARATION(__class__) typedef struct objc_object __class__
#endif

#ifdef __OBJC__
#define CUTE_TYPE_ALIAS(__true_type__, __alias_type__) \
    typedef __true_type__ __alias_type__
#else
#define CUTE_TYPE_ALIAS(__true_type__, __alias_type__) \
    typedef void* __alias_type__
#endif

#endif // !CUTE_MACROS_H_