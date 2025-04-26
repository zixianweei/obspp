cmake_minimum_required(VERSION 3.28)
project(cute)

include(cmake/metal.cmake)

find_package(spdlog CONFIG REQUIRED)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")

add_executable(${PROJECT_NAME} MACOSX_BUNDLE)
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

# set_property(TARGET ${PROJECT_NAME} APPEND_STRING PROPERTY COMPILE_FLAGS "-fobjc-arc")

# target_compile_definitions(
#     ${PROJECT_NAME}
#     PRIVATE
#     HAS_CUTE_LOGGER
# )

target_compile_options(
    ${PROJECT_NAME}
    PRIVATE
    -fsanitize=address
)

target_link_options(
    ${PROJECT_NAME}
    PRIVATE
    -fsanitize=address
)

target_compile_definitions(
    ${PROJECT_NAME}
    PRIVATE
    CUTE_DEBUG
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    spdlog::spdlog
    $<LINK_LIBRARY:FRAMEWORK,Foundation>
    $<LINK_LIBRARY:FRAMEWORK,Metal>
    $<LINK_LIBRARY:FRAMEWORK,QuartzCore>
)

target_sources(
    ${PROJECT_NAME}
    PRIVATE
    main.cpp
    context.mm
    context.h
    tensor.mm
    tensor.h
    op_base.mm
    op_base.h
    op_flip.mm
    op_flip.h
    logger.cpp
    logger.h
)

target_link_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${CMAKE_BINARY_DIR}/opencv/lib
)

function(cv_target_add module)
    target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/opencv/modules/${module}/include
    )
    target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        opencv_${module}
    )
endfunction()

cv_target_add(core)
cv_target_add(dnn)
cv_target_add(imgproc)
cv_target_add(highgui)
cv_target_add(imgcodecs)
cv_target_add(videoio)
cv_target_add(video)
