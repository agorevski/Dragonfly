set(_v41_dir "${CMAKE_CURRENT_LIST_DIR}")
set(_v41_compat "${_v41_dir}/../compat")
set(_v41_inc "${_v41_compat}/include" "${DS41_STRATA_ROOT}/include" "${_v41_dir}")
set(_v41_cpu "${DS41_STRATA_ROOT}/src/kernels/cpu")

# NativeFmt and the pool's scratch buffers are larger than Qwen's. Never link this ABI to strata_kernels_cpu.
add_library(ds41_kernels_cpu STATIC
    "${_v41_compat}/cpu/native_expert.cpp" "${_v41_compat}/cpu/pool.cpp" "${_v41_compat}/cpu/ptq1_0.cpp"
    "${_v41_cpu}/expert.cpp" "${_v41_cpu}/expert_layout.cpp" "${_v41_cpu}/q2_avx2.cpp"
    "${_v41_cpu}/iq_avx512.cpp" "${_v41_cpu}/iq_avx2.cpp" "${_v41_cpu}/kq_avx2.cpp"
    "${_v41_cpu}/kq_avx1.cpp" "${_v41_cpu}/q2_avx1.cpp" "${_v41_cpu}/iq_avx1.cpp"
    "${_v41_cpu}/nvfp4_avx512.cpp" "${_v41_cpu}/nvfp4_avx2.cpp")
target_include_directories(ds41_kernels_cpu PUBLIC ${_v41_inc}
    PRIVATE "${_v41_cpu}" "${DS41_STRATA_ROOT}/third_party/ggml")
target_link_libraries(ds41_kernels_cpu PUBLIC ggml-cpu ggml-base Threads::Threads)
target_compile_definitions(ds41_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
if(MSVC)
    target_compile_options(ds41_kernels_cpu PRIVATE /arch:AVX2)
    set_source_files_properties("${_v41_cpu}/expert.cpp" "${_v41_cpu}/iq_avx512.cpp" "${_v41_cpu}/nvfp4_avx512.cpp"
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    target_compile_options(ds41_kernels_cpu PRIVATE -mavx2 -mfma -mf16c)
    set_source_files_properties("${_v41_cpu}/expert.cpp" "${_v41_cpu}/iq_avx512.cpp" "${_v41_cpu}/nvfp4_avx512.cpp"
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi")
    set_source_files_properties("${_v41_cpu}/q2_avx1.cpp" "${_v41_cpu}/iq_avx1.cpp"
        PROPERTIES COMPILE_OPTIONS "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
endif()

add_library(ds41_dense STATIC "${_v41_dir}/ds41_dense.cpp" "${_v41_dir}/ds41_engram.cpp")
target_include_directories(ds41_dense PUBLIC ${_v41_inc} PRIVATE "${_ds41_ggml_src}")
target_link_libraries(ds41_dense PUBLIC ggml ggml-cpu ggml-base Threads::Threads)

add_library(ds41_moe_cpu STATIC "${_v41_dir}/ds41_moe.cpp")
target_include_directories(ds41_moe_cpu PUBLIC ${_v41_inc})
target_link_libraries(ds41_moe_cpu PUBLIC ds41_kernels_cpu ggml ggml-base)
set(_v41_moe ds41_moe_cpu)

if(STRATA_DS41_CUDA)
    add_library(ds41_kernels_cuda STATIC
        "${_v41_compat}/cuda/iq_kernels.cu" "${_v41_compat}/cuda/native_mmvq.cu" "${_v41_compat}/cuda/pdl.cpp")
    target_include_directories(ds41_kernels_cuda PUBLIC ${_v41_inc}
        PRIVATE "${DS41_STRATA_ROOT}/src/kernels/cuda" "${DS41_STRATA_ROOT}/third_party/ggml")
    target_link_libraries(ds41_kernels_cuda PUBLIC CUDA::cudart)
    set_target_properties(ds41_kernels_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

    add_library(ds41_cache STATIC "${_v41_compat}/core/expert_cache.cpp"
        "${DS41_STRATA_ROOT}/src/core/vmm.cpp" "${DS41_STRATA_ROOT}/src/core/pinned.cu"
        "${DS41_STRATA_ROOT}/src/platform/memory.cpp")
    target_include_directories(ds41_cache PUBLIC ${_v41_inc})
    target_link_libraries(ds41_cache PUBLIC CUDA::cudart Threads::Threads)

    add_library(ds41_transcode STATIC "${_v41_dir}/ds41_transcode.cu")
    target_include_directories(ds41_transcode PUBLIC "${_v41_dir}")
    target_link_libraries(ds41_transcode PUBLIC CUDA::cudart)

    add_library(ds41_moe_cuda STATIC "${_v41_dir}/ds41_moe.cpp")
    target_compile_features(ds41_moe_cuda PUBLIC cxx_std_20)
    target_include_directories(ds41_moe_cuda PUBLIC ${_v41_inc})
    target_compile_definitions(ds41_moe_cuda PRIVATE DS4_MOE_CUDA=1)
    target_link_libraries(ds41_moe_cuda PUBLIC ds41_transcode ds41_cache ds41_kernels_cuda ds41_kernels_cpu ggml)
    # The correctness path uses native grouped experts; MMQ's fork-specific ABI is deliberately not linked.
    set(_v41_moe ds41_moe_cuda)
    target_compile_definitions(ds41_dense PRIVATE DS41_HAVE_CUDA)
    target_link_libraries(ds41_dense PUBLIC ${_ds41_cuda_backend})
endif()

add_executable(ds41_generate "${_v41_dir}/ds41_generate.cpp")
target_link_libraries(ds41_generate PRIVATE ds41_dense ${_v41_moe})
if(STRATA_DS41_CUDA)
    target_compile_definitions(ds41_generate PRIVATE DS41_HAVE_CUDA)
endif()

if(STRATA_DS41_DSPARK)
    target_sources(ds41_generate PRIVATE "${_v41_dir}/ds41_dspark.cpp")
    target_compile_definitions(ds41_generate PRIVATE DS41_HAVE_DSPARK)
endif()

if(STRATA_DS41_BUILD_TESTS)
    # Real-weight tests are built but not silently registered without their input.
    add_executable(test_ds41_geometry "${_v41_dir}/test_ds41_geometry.cpp")
    target_include_directories(test_ds41_geometry PRIVATE ${_v41_inc})
    add_executable(test_ds41_engram_quantized "${_v41_dir}/test_ds41_engram_quantized.cpp" "${_v41_dir}/ds41_engram.cpp")
    target_include_directories(test_ds41_engram_quantized PRIVATE ${_v41_inc})
    target_link_libraries(test_ds41_engram_quantized PRIVATE Threads::Threads)
    add_test(NAME ds41_engram_quantized COMMAND test_ds41_engram_quantized)
    add_executable(test_ds41_ptq1 "${_v41_dir}/test_ds41_ptq1.cpp")
    target_link_libraries(test_ds41_ptq1 PRIVATE ds41_kernels_cpu)
    add_test(NAME ds41_ptq1 COMMAND test_ds41_ptq1)
    add_executable(test_ds41_pool "${_v41_dir}/test_ds41_pool.cpp")
    target_link_libraries(test_ds41_pool PRIVATE ds41_kernels_cpu)
    add_test(NAME ds41_pool COMMAND test_ds41_pool)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        set(_v41_fixture_options)
        if(STRATA_DS41_DSPARK)
            list(APPEND _v41_fixture_options --dspark-built)
        endif()
        add_test(NAME ds41_standard COMMAND "${Python3_EXECUTABLE}" "${_v41_dir}/test_ds41_standard.py"
            --engine "$<TARGET_FILE:ds41_generate>" --work "${CMAKE_CURRENT_BINARY_DIR}/standard-fixture" ${_v41_fixture_options})
        if(STRATA_DS41_CUDA)
            add_test(NAME ds41_standard_cuda COMMAND "${Python3_EXECUTABLE}" "${_v41_dir}/test_ds41_standard.py"
                --engine "$<TARGET_FILE:ds41_generate>" --work "${CMAKE_CURRENT_BINARY_DIR}/standard-cuda-fixture"
                --backend cuda ${_v41_fixture_options})
            set_tests_properties(ds41_standard_cuda PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()
    foreach(_test dense ops)
        add_executable(test_ds41_${_test} "${_v41_dir}/test_ds41_${_test}.cpp")
        target_link_libraries(test_ds41_${_test} PRIVATE ds41_dense ds41_kernels_cpu)
    endforeach()
    if(STRATA_DS41_DSPARK)
        add_executable(test_ds41_dspark "${_v41_dir}/test_ds41_dspark.cpp" "${_v41_dir}/ds41_dspark.cpp")
        target_link_libraries(test_ds41_dspark PRIVATE ds41_dense)
    endif()
    add_executable(test_ds41_moe "${_v41_dir}/test_ds41_moe.cpp")
    target_link_libraries(test_ds41_moe PRIVATE ds41_moe_cpu ggml-cpu)
    if(STRATA_DS41_CUDA)
        add_executable(test_ds41_transcode "${_v41_dir}/test_ds41_transcode.cpp")
        target_link_libraries(test_ds41_transcode PRIVATE ds41_transcode ds41_kernels_cpu ggml CUDA::cudart)
        add_test(NAME ds41_transcode COMMAND test_ds41_transcode)
        add_executable(test_ds41_ptq1_cuda "${_v41_dir}/test_ds41_ptq1_cuda.cpp")
        target_link_libraries(test_ds41_ptq1_cuda PRIVATE ds41_kernels_cuda ds41_kernels_cpu CUDA::cudart)
        add_test(NAME ds41_ptq1_cuda COMMAND test_ds41_ptq1_cuda)
        set_tests_properties(ds41_ptq1_cuda PROPERTIES SKIP_RETURN_CODE 77)
        add_executable(test_ds41_moe_gpu "${_v41_dir}/test_ds41_moe.cpp")
        target_link_libraries(test_ds41_moe_gpu PRIVATE ds41_moe_cuda ggml-cpu)
        add_executable(test_ds41_prom "${_v41_dir}/test_ds41_prom.cpp")
        target_link_libraries(test_ds41_prom PRIVATE ds41_moe_cuda)
    endif()
endif()
