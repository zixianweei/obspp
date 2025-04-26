# https://developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files?language=objc

if (NOT APPLE)
    return()
endif()

set(METAL_CFLAGS -Wall -Wextra -fno-fast-math)
if (WERROR)
    string(APPEND METAL_CFLAGS -Werror)
endif()

function(metal_to_air)
endfunction()

function(air_to_metallib)
    
endfunction()


