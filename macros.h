#ifndef CUTE_MACROS_H_
#define CUTE_MACROS_H_

#define SAFE_RELEASE(__object__) \
    if (__object__ != nil) {     \
        [__object__ release];    \
    }

#endif // !CUTE_MACROS_H_