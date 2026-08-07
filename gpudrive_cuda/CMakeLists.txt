cmake_minimum_required(VERSION 3.24)

project(gpudrive_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(drive_sim
    src/main.cu
    src/drive_sim.cu
)

set_target_properties(drive_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(drive_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
