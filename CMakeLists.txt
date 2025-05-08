cmake_minimum_required(VERSION 3.28)
project(CuteNN)

find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)

include(CTest)
include(GoogleTest)
enable_testing()

find_package(GTest CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# for metal shaders
# include(${CMAKE_CURRENT_SOURCE_DIR}/build/metal.cmake)
# file(GLOB_RECURSE CUTE_METAL_SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/nn/*.metal")
# message(STATUS "CUTE_METAL_SHADERS: ${CUTE_METAL_SHADERS}")
# foreach(CUTE_METAL_SHADER ${CUTE_METAL_SHADERS})
#     cmake_path(GET CUTE_METAL_SHADER STEM TGT_STEM)
#     string(CONCAT TGT_BASIC ${TGT_STEM} "_30.air")
#     list(APPEND AIR_BASIC ${TGT_BASIC})
#     cute_metal_to_air(${CUTE_METAL_SHADER} ${TGT_BASIC} "")
# endforeach()
# cute_air_to_metallib(kernels.metallib ${AIR_BASIC})
# add_custom_target(${PROJECT_NAME}_metallib DEPENDS kernels.metallib)
# set(CUTE_METALLIB_BINARY_DIR ${CMAKE_BINARY_DIR})

add_executable(${PROJECT_NAME})
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

target_compile_definitions(
    ${PROJECT_NAME}
    PRIVATE
    CUTENN_DEBUG
    HAVE_CPUONLY
    HAVE_METAL
)

target_compile_options(
    ${PROJECT_NAME}
    PRIVATE
    -fsanitize=address
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_options(
    ${PROJECT_NAME}
    PRIVATE
    -fsanitize=address
    # -Wl,-sectcreate,__TEXT,metal_basic,${CUTE_METALLIB_BINARY_DIR}/kernels.metallib
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    spdlog::spdlog
    GTest::gtest
    GTest::gmock
    $<LINK_LIBRARY:FRAMEWORK,Foundation>
    $<LINK_LIBRARY:FRAMEWORK,Metal>
    $<LINK_LIBRARY:FRAMEWORK,QuartzCore>
)

# add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_metallib)

target_sources(
    ${PROJECT_NAME}
    PRIVATE
    base/context.hpp
    base/context.mm
    base/logger.cpp
    base/logger.hpp
    base/macros.hpp
    base/tensor.hpp
    base/tensor.mm
    base/types.hpp
    nn/op_base.cpp
    nn/op_base.h
    nn/op_flip_unittest.cpp
    nn/op_flip.cpp
    nn/op_flip.h
    nn/op_softmax_unittest.cpp
    nn/op_softmax.cpp
    nn/op_softmax.hpp
    nn/types.hpp
    testing/cutenn_test_main.cpp
)

gtest_add_tests(TARGET ${PROJECT_NAME})
