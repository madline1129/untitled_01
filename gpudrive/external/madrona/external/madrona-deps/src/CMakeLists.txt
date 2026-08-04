cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)

if (APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
    list(JOIN CMAKE_OSX_ARCHITECTURES "$<SEMICOLON>" EXT_OSX_ARCHITECTURES)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0")
endif()

set(NAME madrona-deps-compile)
project(${NAME})

include(FetchContent)
include(ExternalProject)

get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
include(${PARENT_DIR}/cmake/sys-detect.cmake)

find_package(Python 3.6 COMPONENTS Interpreter)

include(versions.cmake)

if (MADRONA_LINUX)
    set(TOOLCHAIN_OS "linux")
    set(TOOLCHAIN_HASH "${TOOLCHAIN_LINUX_HASH}")

    set(DXC_LINKER_FLAGS "-nostdlib++ ${TOOLCHAIN_PATH}/lib/x86_64-pc-linux-gnu/libc++.a")
elseif (MADRONA_MACOS)
    set(TOOLCHAIN_OS "macos")
    set(TOOLCHAIN_HASH "${TOOLCHAIN_MACOS_HASH}")

    set(DXC_LINKER_FLAGS "-nostdlib++ ${TOOLCHAIN_PATH}/lib/libc++.a")
elseif (MADRONA_WINDOWS)
    set(TOOLCHAIN_OS "windows")
    set(TOOLCHAIN_HASH "${TOOLCHAIN_WINDOWS_HASH}")
endif ()

if (NOT MADRONA_WINDOWS) #FIX
    execute_process(COMMAND uname -m
        OUTPUT_VARIABLE TOOLCHAIN_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    FetchContent_Declare(madrona_toolchain
        URL "https://github.com/shacklettbp/madrona-toolchain/releases/download/${TOOLCHAIN_VERSION}/madrona-toolchain-${TOOLCHAIN_VERSION}-${TOOLCHAIN_OS}-${TOOLCHAIN_ARCH}.tar.xz"
        URL_HASH SHA256=${TOOLCHAIN_HASH}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
    )
    FetchContent_MakeAvailable(madrona_toolchain)
    set(TOOLCHAIN_PATH "${madrona_toolchain_SOURCE_DIR}/toolchain")

    if (MADRONA_MACOS)
        file(GLOB TOOLCHAIN_PATH "${TOOLCHAIN_PATH}/Toolchains/LLVM*.xctoolchain/usr")
    endif()

    set(TOOLCHAIN_C_COMPILER "${TOOLCHAIN_PATH}/bin/clang")
    set(TOOLCHAIN_CXX_COMPILER "${TOOLCHAIN_PATH}/bin/clang++")
else ()
    set(TOOLCHAIN_C_COMPILER "${CMAKE_C_COMPILER}")
    set(TOOLCHAIN_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

if (MADRONA_LINUX)
    if (MADRONA_X64)
        set(TOOLCHAIN_STATIC_CXX_LINKER_FLAGS
            "-nostdlib++ ${TOOLCHAIN_PATH}/lib/x86_64-pc-linux-gnu/libc++.a")
    else ()
        message(FATAL_ERROR "ARM Linux deps not supported")
    endif ()
elseif (MADRONA_MACOS)
    set(TOOLCHAIN_STATIC_CXX_LINKER_FLAGS
        "-nostdlib++ ${TOOLCHAIN_PATH}/lib/libc++.a")
elseif (MADRONA_WINDOWS)
    set(TOOLCHAIN_STATIC_CXX_LINKER_FLAGS "")
endif ()

if (MADRONA_LINUX)
    set (SDL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/SDL")
    set (SDL_INSTALL "${SDL_PREFIX}/install")
    
    list(APPEND SDL_ARGS
        "-DCMAKE_C_COMPILER=${TOOLCHAIN_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER=${TOOLCHAIN_CXX_COMPILER}"
        "-DCMAKE_INSTALL_PREFIX=${SDL_INSTALL}"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DSDL_SHARED=ON"
        "-DSDL_STATIC=OFF"
        "-DSDL_ASSERTIONS=release"
        "-DSDL_WAYLAND=OFF"
        "-DSDL_KMSDRM=OFF"
        "-DSDL_OPENGLES=OFF"
        "-DSDL_DISKAUDIO=OFF"
        "-DSDL_X11=ON"
        "-DSDL_OPENGL=ON"
        "-DSDL_VULKAN=ON"
    )

    ExternalProject_Add(SDL
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG "${SDL_VERSION}"
        GIT_SHALLOW ON
        PREFIX "${SDL_PREFIX}"
        INSTALL_DIR "${SDL_INSTALL}"
        CMAKE_ARGS ${SDL_ARGS}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    install(DIRECTORY "${SDL_INSTALL}/include/SDL2"
        DESTINATION SDL/include
    )
    
    install(DIRECTORY ${SDL_INSTALL}/lib/
        DESTINATION SDL/lib
        FILES_MATCHING
            PATTERN *
            PATTERN *pkgconfig* EXCLUDE
    )
endif()

set (VK_HDRS_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/vk_hdrs")
set (VK_HDRS_INSTALL "${VK_HDRS_PREFIX}/install")

list(APPEND VK_HDRS_ARGS
    "-DCMAKE_C_COMPILER=${TOOLCHAIN_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${TOOLCHAIN_CXX_COMPILER}"
    "-DCMAKE_INSTALL_PREFIX=${VK_HDRS_INSTALL}"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CONFIGURATION_TYPES=Release"
    "-DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}"
)

if (MADRONA_MACOS OR MADRONA_IOS)
    list(APPEND VK_HDRS_ARGS
        -DCMAKE_OSX_ARCHITECTURES=${EXT_OSX_ARCHITECTURES}
        -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
    )
endif()

ExternalProject_Add(VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG vulkan-sdk-${VULKAN_HEADER_VERSION}
    PREFIX "${VK_HDRS_PREFIX}"
    INSTALL_DIR "${VK_HDRS_INSTALL}"
    CMAKE_ARGS ${VK_HDRS_ARGS}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

install(DIRECTORY ${VK_HDRS_INSTALL}/include/
    DESTINATION vk_hdrs/include/
)

set(VK_LOADER_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/vk_loader")
set(VK_LOADER_INSTALL "${VK_LOADER_PREFIX}/install")

list(APPEND VK_LOADER_ARGS
    "-DCMAKE_C_COMPILER=${TOOLCHAIN_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${TOOLCHAIN_CXX_COMPILER}"
    "-DCMAKE_INSTALL_PREFIX=${VK_LOADER_INSTALL}"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CONFIGURATION_TYPES=Release"
    "-DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}"
    "-DUPDATE_DEPS=ON"
)

if (MADRONA_MACOS OR MADRONA_IOS)
    list(APPEND VK_LOADER_ARGS
        -DCMAKE_OSX_ARCHITECTURES=${EXT_OSX_ARCHITECTURES}
        -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
    )
endif()

ExternalProject_Add(VulkanLoader
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Loader.git
    GIT_TAG vulkan-sdk-${VULKAN_LOADER_VERSION}
    PREFIX "${VK_LOADER_PREFIX}"
    INSTALL_DIR "${VK_LOADER_INSTALL}"
    CMAKE_ARGS ${VK_LOADER_ARGS}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

if (MADRONA_WINDOWS)
    install(DIRECTORY ${VK_LOADER_INSTALL}/bin/
        DESTINATION vk_loader/lib
        FILES_MATCHING
            PATTERN *dll
    )
else()
    install(DIRECTORY ${VK_LOADER_INSTALL}/lib/
        DESTINATION vk_loader/lib
        FILES_MATCHING
            PATTERN *
            PATTERN *pkgconfig* EXCLUDE
    )
endif()

if (MADRONA_MACOS)
    set (MOLTENVK_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/moltenvk")

    ExternalProject_Add(MoltenVK
        GIT_REPOSITORY https://github.com/KhronosGroup/MoltenVK.git
        GIT_TAG ${MOLTENVK_VERSION}
        PREFIX ${MOLTENVK_PREFIX}
        STAMP_DIR ${MOLTENVK_PREFIX}/stamps
        SOURCE_DIR ${MOLTENVK_PREFIX}/repo
        BINARY_DIR ${MOLTENVK_PREFIX}/repo
        CONFIGURE_COMMAND ""
        BUILD_COMMAND
            ./fetchDependencies --macos
            COMMAND make MVK_CONFIG_LOG_LEVEL=1 MVK_HIDE_VULKAN_SYMBOLS=1 macos
        INSTALL_COMMAND ""
    )

    install(DIRECTORY ${MOLTENVK_PREFIX}/repo/Package/Release/MoltenVK/dynamic/dylib/macOS/
        DESTINATION moltenvk/lib/
    )

    install(DIRECTORY ${MOLTENVK_PREFIX}/repo/Package/Release/MoltenVK/include/MoltenVK
        DESTINATION moltenvk/include/
    )

    FetchContent_Declare(metal-cpp
        URL "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS13_iOS16.zip"
        URL_HASH SHA256=6f741894229e9c750add1afc3797274fc008c7507e2ae726370c17c34b7c6a68
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(metal-cpp)

    install(DIRECTORY ${metal-cpp_SOURCE_DIR}/Foundation
        DESTINATION metal-cpp/include
    )
    install(DIRECTORY ${metal-cpp_SOURCE_DIR}/Metal
        DESTINATION metal-cpp/include
    )
    install(DIRECTORY ${metal-cpp_SOURCE_DIR}/QuartzCore
        DESTINATION metal-cpp/include
    )
    install(FILES ${metal-cpp_SOURCE_DIR}/LICENSE.txt
        DESTINATION metal-cpp
    )

    # Want metal-cpp-extensions from this example
    # (why is it not distributed separately??)
    FetchContent_Declare(metal-cpp-extensions
        URL "https://developer.apple.com/metal/LearnMetalCPP.zip"
        URL_HASH SHA256=a709f3c0b532d5b9e3a5db3da3a35f3b783af27eb50f23a711115c02f86a256d
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
    )
    FetchContent_MakeAvailable(metal-cpp-extensions)

    set(metal_cpp_ext_dir "${metal-cpp-extensions_SOURCE_DIR}/metal-cpp-extensions")
    install(DIRECTORY "${metal_cpp_ext_dir}/AppKit"
        DESTINATION metal-cpp-extensions/include
    )
    install(DIRECTORY "${metal_cpp_ext_dir}/MetalKit"
        DESTINATION metal-cpp-extensions/include
    )
endif ()

set(DXC_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/dxc")
set(DXC_SRC "${DXC_PREFIX}/src")
set(DXC_BUILD "${DXC_PREFIX}/build")
set(DXC_INSTALL "${DXC_PREFIX}/install")

list(APPEND DXC_CMAKE_ARGS
    "-DCMAKE_INSTALL_PREFIX=${DXC_INSTALL}"
    "-DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}"
    "-DCMAKE_C_COMPILER=${TOOLCHAIN_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${TOOLCHAIN_CXX_COMPILER}"
    "-DCMAKE_CONFIGURATION_TYPES=Release"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_OSX_ARCHITECTURES=${EXT_OSX_ARCHITECTURES}"
    "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
    "-DCMAKE_EXE_LINKER_FLAGS=${TOOLCHAIN_STATIC_CXX_LINKER_FLAGS}"
    "-DCMAKE_SHARED_LINKER_FLAGS=${TOOLCHAIN_STATIC_CXX_LINKER_FLAGS}"
    "-DCMAKE_MODULE_LINKER_FLAGS=${TOOLCHAIN_STATIC_CXX_LINKER_FLAGS}"
    "-DLLVM_ENABLE_ZLIB=OFF"
    "-DENABLE_SPIRV_CODEGEN=ON"
    "-C ${DXC_SRC}/cmake/caches/PredefinedParams.cmake"
)

if (MADRONA_MACOS)
    # There is an annoying issue here: llvm-ar does not work with universal
    # binaries, because of this CMake uses system ar by default on macos.
    # However, it still tries to use llvm-ranlib by default, which then
    # breaks universal binaries. One option is to force system ranlib on macos
    # The better option is to do what LLVM itself does, which is to just use
    # libtool on macos for building static libraries since it is more
    # optimized anyway: https://reviews.llvm.org/D19611. llvm-libtool
    # correctly handles universal binaries
    list(APPEND DXC_CMAKE_ARGS
        "-DCMAKE_CXX_CREATE_STATIC_LIBRARY=\"${TOOLCHAIN_PATH}/bin/llvm-libtool-darwin\" -static -no_warning_for_no_symbols -o <TARGET> <LINK_FLAGS> <OBJECTS>"
        "-DCMAKE_RANLIB=\"\""
    )
endif()

ExternalProject_Add(dxc
    GIT_REPOSITORY https://github.com/microsoft/DirectXShaderCompiler.git
    GIT_TAG "${DXC_VERSION}"
    PREFIX "${DXC_PREFIX}"
    SOURCE_DIR "${DXC_SRC}"
    STAMP_DIR "${DXC_PREFIX}/stamps"
    BINARY_DIR "${DXC_BUILD}"
    INSTALL_DIR "${DXC_INSTALL}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_GENERATOR Ninja
    INSTALL_COMMAND ninja -C "${DXC_BUILD}" install-distribution
    CMAKE_ARGS ${DXC_CMAKE_ARGS}
)

if (MADRONA_WINDOWS)
    find_package(Python 3.9 COMPONENTS Interpreter REQUIRED)

    ExternalProject_Add_Step(dxc download_taef
        COMMAND ${CMAKE_COMMAND} -E env HLSL_SRC_DIR=${DXC_SRC} ${Python_EXECUTABLE} "${DXC_SRC}/utils/hct/hctgettaef.py"
        DEPENDEES download update patch
        DEPENDERS configure
    )

    # For some reason dxcompiler.lib is left out of install-distribution
    install(FILES ${DXC_BUILD}/lib/dxcompiler.lib
        DESTINATION dxc/lib/
    )
else ()
    install(DIRECTORY ${DXC_INSTALL}/lib
        DESTINATION dxc/
        USE_SOURCE_PERMISSIONS
    )
endif ()

install(DIRECTORY ${DXC_INSTALL}/include
    DESTINATION dxc/
    USE_SOURCE_PERMISSIONS
)

install(DIRECTORY ${DXC_INSTALL}/bin
    DESTINATION dxc/
    USE_SOURCE_PERMISSIONS
)
