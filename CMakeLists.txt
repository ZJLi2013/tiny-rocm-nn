# Copyright (c) 2020-2023, NVIDIA CORPORATION.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.18)
#for debug
# set(CMAKE_CXX_FLAGS_DEBUG "-O0")
# set(CMAKE_HIP_FLAGS_DEBUG "-O0")

set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc) 

project(
	tiny-rocm-nn
	VERSION 1.0
	DESCRIPTION "Lightning fast & tiny C++/ROCM neural network framework"
	LANGUAGES CXX HIP
)

option(TCNN_BUILD_BENCHMARK "Build tiny-rocm-nn example benchmark?" OFF)
option(TCNN_BUILD_EXAMPLES "Build tiny-rocm-nn example applications?" ON)

###############################################################################
# Build type and C++ compiler setup
###############################################################################

# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if (APPLE)
	set(CMAKE_MACOSX_RPATH ON)
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# Setup ROCm dependencies  
###############################################################################
# Find ROCm packages
find_package(hip REQUIRED)
find_package(hiprtc REQUIRED )
find_package(rocblas REQUIRED)
set(CMAKE_PREFIX_PATH "/opt/rocm/include/rocwmma")
# Could not find a package configuration file provided by "rocwmma" with rocwmmaConfig.cmake
# find_package(rocwmma REQUIRED)

# Include directories
include_directories(${HIP_INCLUDE_DIRS})
include_directories(${rocblas_INCLUDE_DIRS})
include_directories(${rocwmma_INCLUDE_DIRS})

###############################################################################
# Dependencies
###############################################################################

if (NOT MSVC)
	set(ROCM_TOOLKIT_ROOT_DIR /opt/rocm)
endif()

set(BUILD_SHARED_LIBS OFF)
add_subdirectory("dependencies/fmt")

###############################################################################
# tiny-rocm-nn library, samples, and benchmarks
###############################################################################
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

set(TCNN_SOURCES
	src/common_host.cpp
	src/cpp_api.cpp
#	src/cutlass_mlp.cpp
	src/encoding.cpp
    src/fully_fused_mlp.cpp 
	src/loss.cpp
	src/network.cpp
	src/object.cpp
	src/optimizer.cpp
	src/reduce_sum.cpp
)

add_library(tiny-rocm-nn STATIC ${TCNN_SOURCES})
target_compile_definitions(tiny-rocm-nn PUBLIC ${TCNN_DEFINITIONS})
target_include_directories(tiny-rocm-nn PUBLIC
	"include"
	"dependencies"
	"dependencies/fmt/include"
)
target_link_libraries(tiny-rocm-nn PUBLIC hip::device hip::host roc::rocblas fmt)

if (TCNN_BUILD_EXAMPLES)
	add_subdirectory("samples")
endif()
if (TCNN_BUILD_BENCHMARK)
	add_subdirectory("benchmarks/image")
endif()
