add_executable(fused_unfused_test fused_unfused_test.cpp)

target_link_libraries(fused_unfused_test PRIVATE tiny-rocm-nn)

# Add AMD GPU architecture flags
target_compile_options(fused_unfused_test PRIVATE "--offload-arch=gfx90a")
