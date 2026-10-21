# Included separately so the multi-GPU prototype never changes the generator link path.
add_library(ds41_mgpu STATIC "${_v41_dir}/ds41_mgpu.cpp")
target_include_directories(ds41_mgpu PUBLIC ${_v41_inc})
target_compile_features(ds41_mgpu PUBLIC cxx_std_17)

if(STRATA_DS41_CUDA)
    add_library(ds41_mgpu_cuda STATIC "${_v41_dir}/ds41_mgpu.cpp")
    target_include_directories(ds41_mgpu_cuda PUBLIC ${_v41_inc})
    target_compile_features(ds41_mgpu_cuda PUBLIC cxx_std_17)
    target_compile_definitions(ds41_mgpu_cuda PRIVATE DS41_MGPU_CUDA=1)
    target_link_libraries(ds41_mgpu_cuda PUBLIC ds41_kernels_cuda CUDA::cudart)
endif()

if(STRATA_DS41_BUILD_TESTS)
    add_executable(test_ds41_mgpu "${_v41_dir}/test_ds41_mgpu.cpp")
    target_link_libraries(test_ds41_mgpu PRIVATE ds41_mgpu ds41_moe_cpu)
    add_test(NAME ds41_mgpu_plan COMMAND test_ds41_mgpu)
    if(STRATA_DS41_CUDA)
        add_executable(test_ds41_mgpu_cuda "${_v41_dir}/test_ds41_mgpu_cuda.cpp")
        target_link_libraries(test_ds41_mgpu_cuda PRIVATE ds41_mgpu_cuda ds41_moe_cpu ggml-cpu)
        # GPU tests are explicit, never automatic: --devices needs a granted lease.
    endif()
endif()
