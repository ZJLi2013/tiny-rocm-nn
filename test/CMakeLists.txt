add_executable(fused_unfused_test fused_unfused_test.cpp)

# Add the necessary include directories for the headers.
target_include_directories(fused_unfused_test PRIVATE
    ../include
    ../dependencies
)

# Link against the main library, which provides all other symbols needed by the included .cpp file.
target_link_libraries(fused_unfused_test PRIVATE tiny-rocm-nn)

# Add AMD GPU architecture flags
target_compile_options(fused_unfused_test PRIVATE "--offload-arch=gfx942")
