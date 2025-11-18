add_executable(fused_unfused_test
    fused_unfused_test.cpp
    ../src/fully_fused_mlp.cpp
)

# By compiling fully_fused_mlp.cpp directly, we no longer need to link against
# the full tiny-rocm-nn library for this specific test, which can cause
# duplicate symbol errors. We link against its dependencies instead.
target_link_libraries(fused_unfused_test PRIVATE hip::device hip::host roc::rocblas roc::hipblas fmt)

# Add AMD GPU architecture flags
target_compile_options(fused_unfused_test PRIVATE "--offload-arch=gfx90a")
