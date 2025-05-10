cmake_minimum_required(VERSION 3.28)
project(MIXTURE)

find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)

find_package(GTest CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

include(CTest)
include(GoogleTest)
enable_testing()

set(TARGET_NAME "cutenn" CACHE INTERNAL "")
add_library(${TARGET_NAME} SHARED)

set_target_properties(
    ${TARGET_NAME}
    PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE
    CUTENN_METAL_DEBUG
)

target_compile_options(
    ${TARGET_NAME}
    PRIVATE
    -fsanitize=address
)

target_link_options(
    ${TARGET_NAME}
    PRIVATE
    -fsanitize=address
)

target_include_directories(
    ${TARGET_NAME}
    PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(
    ${TARGET_NAME}
    PUBLIC
    spdlog::spdlog
    PRIVATE
    $<LINK_LIBRARY:FRAMEWORK,Foundation>
    $<LINK_LIBRARY:FRAMEWORK,Metal>
    $<LINK_LIBRARY:FRAMEWORK,QuartzCore>
)

target_sources(
    ${TARGET_NAME}
    PRIVATE
    base/context.h
    base/context.mm
    base/logger.cpp
    base/logger.h
    base/macros.h
    base/tensor.h
    base/tensor.mm
    base/types.h
    nn/op_base.h
    nn/op_base.mm
    nn/op_flip.h
    nn/op_flip.mm
    nn/op_softmax.h
    nn/op_softmax.mm
    nn/types.h
)

set(TARGET_UNITTEST_NAME "${TARGET_NAME}_unittest" CACHE INTERNAL "")
add_executable(${TARGET_UNITTEST_NAME})
gtest_add_tests(TARGET ${TARGET_UNITTEST_NAME})

set_target_properties(
    ${TARGET_UNITTEST_NAME}
    PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

target_compile_options(
    ${TARGET_UNITTEST_NAME}
    PRIVATE
    -fsanitize=address
)

target_link_options(
    ${TARGET_UNITTEST_NAME}
    PRIVATE
    -fsanitize=address
)

target_include_directories(
    ${TARGET_UNITTEST_NAME}
    PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(
    ${TARGET_UNITTEST_NAME}
    PRIVATE
    ${TARGET_NAME}
    GTest::gtest
    GTest::gmock
)

target_sources(
    ${TARGET_UNITTEST_NAME}
    PRIVATE
    testing/op_flip_unittest.cpp
    testing/op_softmax_unittest.cpp
    testing/cutenn_test_main.cpp
)
