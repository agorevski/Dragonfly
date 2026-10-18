cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(strata_ds41 LANGUAGES C CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

get_filename_component(DS41_STRATA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
option(STRATA_DS41_CUDA "Build the isolated DeepSeek-V4.1 CUDA engine" OFF)
option(STRATA_DS41_CUDA_GRAPHS "Enable stock CUDA graph capture in the isolated DS41 backend" OFF)
set(STRATA_DS41_CUDA_BUILD_JOBS "8" CACHE STRING "Parallel jobs for the isolated ggml CUDA build")
option(STRATA_DS41_BUILD_TESTS "Build DeepSeek-V4.1 tests (including CPU-only fixtures)" ON)
option(STRATA_DS41_DSPARK "Build the optional DSpark draft stack" OFF)
set(STRATA_DS41_GGML_DIR "" CACHE PATH "llama.cpp checkout for standalone DS41 builds")

if(STRATA_DS41_CUDA)
    if(STRATA_ENABLE_HIP OR STRATA_ENABLE_SYCL OR STRATA_HIP_GFX906)
        message(FATAL_ERROR "DeepSeek-V4.1 currently supports CPU and NVIDIA CUDA only")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(NOT TARGET ggml-cpu)
    enable_language(C)
    set(BUILD_SHARED_LIBS OFF)
    set(GGML_STATIC ON)
    set(GGML_CUDA OFF)
    set(GGML_OPENMP OFF)
    set(GGML_BACKEND_DL OFF)
    set(GGML_BUILD_TESTS OFF)
    set(GGML_BUILD_EXAMPLES OFF)
    if(STRATA_DS41_GGML_DIR)
        set(_ds41_llama "${STRATA_DS41_GGML_DIR}")
    elseif(STRATA_GGML_DIR)
        set(_ds41_llama "${STRATA_GGML_DIR}")
    else()
        include(FetchContent)
        FetchContent_Declare(ds41_llamacpp
            URL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/3cf03257f219afbe7334045ff7c6a06ac68c627d
            URL_HASH SHA256=c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_GetProperties(ds41_llamacpp)
        if(NOT ds41_llamacpp_POPULATED)
            FetchContent_Populate(ds41_llamacpp)
        endif()
        set(_ds41_llama "${ds41_llamacpp_SOURCE_DIR}")
    endif()
    add_subdirectory("${_ds41_llama}/ggml" "${CMAKE_CURRENT_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
endif()

get_target_property(_ds41_ggml_src ggml-base SOURCE_DIR)
if(STRATA_DS41_CUDA)
    if(TARGET ggml-cuda)
        if(STRATA_DS41_CUDA_GRAPHS AND NOT GGML_CUDA_GRAPHS)
            message(FATAL_ERROR "STRATA_DS41_CUDA_GRAPHS requires the existing ggml-cuda target to use GGML_CUDA_GRAPHS=ON")
        endif()
        set(_ds41_cuda_backend ggml-cuda)
    else()
        # ggml_add_backend_library links the backend to the global ggml target. Build the stock backend
        # separately instead: neither ggml's registry/defines nor upstream's kernel targets are changed.
        include(ExternalProject)
        set(_ds41_cuda_build "${CMAKE_CURRENT_BINARY_DIR}/ds41-ggml-cuda")
        set(_ds41_cuda_archive "${_ds41_cuda_build}/ggml/src/ggml-cuda/${CMAKE_STATIC_LIBRARY_PREFIX}ggml-cuda${CMAKE_STATIC_LIBRARY_SUFFIX}")
        ExternalProject_Add(ds41_ggml_cuda_build
            SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake/cuda-backend"
            BINARY_DIR "${_ds41_cuda_build}"
            DOWNLOAD_COMMAND ""
            CMAKE_ARGS
                "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
                "-DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}"
                "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
                "-DCUDAToolkit_ROOT=${CUDAToolkit_BIN_DIR}/.."
                "-DDS41_GGML_SOURCE=${_ds41_ggml_src}/.."
                -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=OFF -DGGML_CUDA=ON
                "-DGGML_CUDA_GRAPHS=${STRATA_DS41_CUDA_GRAPHS}"
                -DGGML_BACKEND_DL=OFF -DGGML_OPENMP=OFF -DGGML_CUDA_NCCL=OFF
                -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
            BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target ggml-cuda --parallel "${STRATA_DS41_CUDA_BUILD_JOBS}"
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS "${_ds41_cuda_archive}")
        add_library(ds41_ggml_cuda STATIC IMPORTED)
        add_dependencies(ds41_ggml_cuda ds41_ggml_cuda_build)
        set_target_properties(ds41_ggml_cuda PROPERTIES
            IMPORTED_LOCATION "${_ds41_cuda_archive}"
            INTERFACE_INCLUDE_DIRECTORIES "${_ds41_ggml_src}/../include")
        target_link_libraries(ds41_ggml_cuda INTERFACE ggml-base CUDA::cudart CUDA::cublas CUDA::cuda_driver)
        set(_ds41_cuda_backend ds41_ggml_cuda)
    endif()
endif()

find_package(Threads REQUIRED)
if(STRATA_DS41_BUILD_TESTS)
    enable_testing()
endif()
include("${CMAKE_CURRENT_LIST_DIR}/engine/v41_engine.cmake")
