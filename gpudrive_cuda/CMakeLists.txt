cmake_minimum_required(VERSION 3.24)

project(gpudrive_cuda LANGUAGES CXX CUDA)

option(BUILD_TORCH_BINDINGS "Build the CUDA Tensor Python extension" OFF)

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

add_library(runtime_scene STATIC
    src/runtime_scene.cpp
)
target_include_directories(runtime_scene PUBLIC src)
target_link_libraries(runtime_scene PUBLIC nlohmann_json::nlohmann_json)
set_target_properties(runtime_scene PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(runtime_scene PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)

add_library(drive_simulator STATIC
    src/drive_sim.cu
)
target_include_directories(drive_simulator PUBLIC src)
target_link_libraries(drive_simulator PUBLIC runtime_scene)
set_target_properties(drive_simulator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(drive_simulator PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--expt-relaxed-constexpr>"
)

add_executable(drive_sim_cli
    src/main.cu
)
target_link_libraries(drive_sim_cli PRIVATE drive_simulator runtime_scene)
set_target_properties(drive_sim_cli PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(BUILD_TORCH_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(Torch REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    find_library(TORCH_PYTHON_LIBRARY torch_python
        PATHS "${TORCH_INSTALL_PREFIX}/lib"
        NO_DEFAULT_PATH
        REQUIRED
    )
    separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

    # Torch's prebuilt wheels use a specific libstdc++ ABI.  Every C++/CUDA
    # target linked into the Python module must use the same ABI, otherwise
    # filesystem/string symbols from the static simulator libraries remain
    # unresolved when Python imports the extension.
    target_compile_options(runtime_scene PRIVATE ${TORCH_CXX_FLAGS_LIST})
    target_compile_options(drive_simulator PRIVATE ${TORCH_CXX_FLAGS_LIST})
    target_compile_options(drive_sim_cli PRIVATE ${TORCH_CXX_FLAGS_LIST})

    pybind11_add_module(gpudrive_cuda_torch MODULE
        src/torch_bindings.cu
    )
    target_link_libraries(gpudrive_cuda_torch PRIVATE
        drive_simulator
        runtime_scene
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARY}
    )
    target_compile_definitions(gpudrive_cuda_torch PRIVATE
        TORCH_EXTENSION_NAME=gpudrive_cuda_torch
    )
    target_compile_options(gpudrive_cuda_torch PRIVATE
        ${TORCH_CXX_FLAGS_LIST}
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
    )
    set_target_properties(gpudrive_cuda_torch PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    )
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(runtime_loader_test tests/runtime_loader_test.cpp)
    target_link_libraries(runtime_loader_test PRIVATE runtime_scene)

    add_executable(simulator_integration_test tests/simulator_integration_test.cu)
    target_link_libraries(simulator_integration_test PRIVATE drive_simulator runtime_scene)
    set_target_properties(simulator_integration_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    if(BUILD_TORCH_BINDINGS)
        target_compile_options(runtime_loader_test PRIVATE ${TORCH_CXX_FLAGS_LIST})
        target_compile_options(simulator_integration_test PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
endif()
