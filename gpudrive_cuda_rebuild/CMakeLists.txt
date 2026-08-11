cmake_minimum_required(VERSION 3.24)

project(gpudrive_cuda_rebuild LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../gpudrive/external/json"
    "${CMAKE_CURRENT_BINARY_DIR}/nlohmann_json"
    EXCLUDE_FROM_ALL
)

# RuntimeScenario 的 CPU 加载器独立编译，方便单独测试数据读取。
add_library(runtime_scene STATIC
    src/runtime_scene.cpp
)
target_include_directories(runtime_scene PUBLIC src)
target_link_libraries(runtime_scene PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(runtime_scene PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)

add_executable(drive_sim_base
    src/main.cu
    src/drive_sim.cu
)

target_link_libraries(drive_sim_base PRIVATE runtime_scene)

set_target_properties(drive_sim_base PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(drive_sim_base PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>"
)
