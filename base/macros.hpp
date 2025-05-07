#ifndef CUTENN_BASE_MACROS_HPP_
#define CUTENN_BASE_MACROS_HPP_

#if !__has_feature(objc_arc)
#define CUTENN_SAFE_RELEASE(__object__)                                        \
  if (__object__ != nil) {                                                     \
    [__object__ release];                                                      \
    __object__ = nil;                                                          \
  }
#else
#define CUTENN_SAFE_RELEASE(__object__) __object__ = nil
#endif

#ifdef __OBJC__
#define CUTENN_OBJC_FORWARD_DECLARATION(__class__) @class __class__
#else
#define CUTENN_OBJC_FORWARD_DECLARATION(__class__)                             \
  typedef struct objc_object __class__
#endif

#ifdef __OBJC__
#define CUTENN_TYPE_ALIAS(__true_type__, __alias_type__)                       \
  typedef __true_type__ __alias_type__
#else
#define CUTENN_TYPE_ALIAS(__true_type__, __alias_type__)                       \
  typedef void *__alias_type__
#endif

#endif // !CUTENN_BASE_MACROS_HPP_