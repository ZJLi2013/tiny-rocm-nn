add_executable(fused_unfused_test fused_unfused_test.cpp)

# Add the necessary include directories for the headers.
target_include_directories(fused_unfused_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/dependencies
)

# Link against the main library
target_link_libraries(fused_unfused_test PRIVATE tiny-rocm-nn)

# Add AMD GPU architecture flags (match the main library settings)
target_compile_options(fused_unfused_test PRIVATE "--offload-arch=gfx942")

# Ensure C++14 standard (match the main library)
set_target_properties(fused_unfused_test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# =======================
# fused_unfused_layers_test
# =======================
add_executable(fused_unfused_layers_test fused_unfused_layers_test.cpp)

# Add the necessary include directories for the headers.
target_include_directories(fused_unfused_layers_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/dependencies
)

# Link against the main library
target_link_libraries(fused_unfused_layers_test PRIVATE tiny-rocm-nn)

# Add AMD GPU architecture flags (match the main library settings)
target_compile_options(fused_unfused_layers_test PRIVATE "--offload-arch=gfx942")

# Ensure C++14 standard (match the main library)
set_target_properties(fused_unfused_layers_test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
