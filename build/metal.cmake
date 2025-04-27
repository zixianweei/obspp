# Reference:
# https://developer.apple.com/documentation/metal/building-a-shader-library-by-precompiling-source-files?language=objc
# https://github.com/pytorch/pytorch/blob/main/cmake/Metal.cmake

if (NOT APPLE)
    return()
endif()

set(METAL_CFLAGS -Wall -Wextra -fno-fast-math)
if (WERROR)
    string(APPEND METAL_CFLAGS -Werror)
endif()

function(cute_metal_to_air SRC TGT FLAGS)
    add_custom_command(
        COMMAND xcrun metal -c ${SRC} -I ${CMAKE_SOURCE_DIR} -o ${TGT} ${FLAGS} ${METAL_CFLAGS}
        DEPENDS ${SRC}
        OUTPUT ${TGT}
        COMMENT "Compiling ${SRC} to ${TGT}"
        VERBATIM)
endfunction()

function(cute_air_to_metallib TGT OBJS)
    set(_OBJECTS ${OBJS} ${ARGN})
    add_custom_command(
        COMMAND xcrun metallib -o ${TGT} ${_OBJECTS}
        DEPENDS ${_OBJECTS}
        OUTPUT ${TGT}
        COMMENT "Linking ${TGT}"
        VERBATIM)
endfunction()
