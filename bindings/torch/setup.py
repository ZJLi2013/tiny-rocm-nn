import os
from setuptools import setup
import sys
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Find version of tiny-rocm-nn by scraping CMakeLists.txt
with open(os.path.join(ROOT_DIR, "CMakeLists.txt"), "r") as cmakelists:
	for line in cmakelists.readlines():
		if line.strip().startswith("VERSION"):
			VERSION = line.split("VERSION")[-1].strip()
			break

print(f"Building PyTorch extension for tiny-rocm-nn version {VERSION}")

# ---------------------------------------------------------------------------
# ROCm / HIP setup
# ---------------------------------------------------------------------------
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
HIPCC = os.path.join(ROCM_PATH, "bin", "hipcc")

if not os.path.isfile(HIPCC):
	raise EnvironmentError(
		f"hipcc not found at {HIPCC}. "
		"Set ROCM_PATH to your ROCm installation directory."
	)

os.environ["CXX"] = HIPCC
os.environ["CC"] = HIPCC

rocm_arch = os.environ.get("PYTORCH_ROCM_ARCH", "gfx942")
print(f"Targeting ROCm GPU architecture: {rocm_arch}")

# ---------------------------------------------------------------------------
# Optional: build without neural networks
# ---------------------------------------------------------------------------
include_networks = True
if "--no-networks" in sys.argv:
	include_networks = False
	sys.argv.remove("--no-networks")
	print("Building >> without << neural networks (just the input encodings)")

cpp_standard = 17
print(f"Targeting C++ standard {cpp_standard}")

# ---------------------------------------------------------------------------
# Compiler flags (hipcc)
# ---------------------------------------------------------------------------
base_cflags = [
	f"-std=c++{cpp_standard}",
	"-fPIC",
	"-O3",
	f"--offload-arch={rocm_arch}",
	"-D__HIP_PLATFORM_AMD__",
	"-DUSE_ROCM",
	"-Wno-float-conversion",
	"-fno-strict-aliasing",
	"-fno-gpu-rdc",
	"-munsafe-fp-atomics",
]

os.environ["TORCH_CUDA_ARCH_LIST"] = ""

# ---------------------------------------------------------------------------
# Source files & definitions
# ---------------------------------------------------------------------------
bindings_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(bindings_dir, "../.."))

base_definitions = [
	"-DTCNN_PARAMS_UNALIGNED",
	"-DTCNN_MIN_GPU_ARCH=75",
]

base_source_files = [
	"tinycudann/bindings.cpp",
	"../../dependencies/fmt/src/format.cc",
	"../../dependencies/fmt/src/os.cc",
	"../../src/cpp_api.cpp",
	"../../src/common_host.cpp",
	"../../src/encoding.cpp",
	"../../src/object.cpp",
]

if include_networks:
	base_source_files += [
		"../../src/network.cpp",
		"../../src/fully_fused_mlp.cpp",
	]
else:
	base_definitions.append("-DTCNN_NO_NETWORKS")

# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------
rocm_include = os.path.join(ROCM_PATH, "include")
cflags = base_cflags + base_definitions

ext = CppExtension(
	name="tinycudann_bindings._75_C",
	sources=base_source_files,
	include_dirs=[
		f"{root_dir}/include",
		f"{root_dir}/dependencies",
		f"{root_dir}/dependencies/fmt/include",
		rocm_include,
		os.path.join(rocm_include, "rocwmma"),
		os.path.join(rocm_include, "hipblas"),
		os.path.join(rocm_include, "rocblas"),
	],
	extra_compile_args={"cxx": cflags},
	libraries=["amdhip64", "hipblas", "rocblas"],
	library_dirs=[os.path.join(ROCM_PATH, "lib")],
)

# ---------------------------------------------------------------------------
# Package
# ---------------------------------------------------------------------------
setup(
	name="tinycudann",
	version=VERSION,
	description="tiny-rocm-nn extension for PyTorch (ROCm/HIP)",
	long_description="tiny-rocm-nn extension for PyTorch (ROCm/HIP)",
	classifiers=[
		"Development Status :: 4 - Beta",
		"Environment :: GPU :: AMD ROCm",
		"License :: BSD 3-Clause",
		"Programming Language :: C++",
		"Programming Language :: Python :: 3 :: Only",
		"Topic :: Multimedia :: Graphics",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"Topic :: Scientific/Engineering :: Image Processing",
	],
	keywords="PyTorch,ROCm,HIP,rocwmma,machine learning",
	url="https://github.com/nvlabs/tiny-cuda-nn",
	author="Thomas Müller, Jacob Munkberg, Jon Hasselgren, Or Perel",
	author_email="tmueller@nvidia.com, jmunkberg@nvidia.com, jhasselgren@nvidia.com, operel@nvidia.com",
	maintainer="Thomas Müller",
	maintainer_email="tmueller@nvidia.com",
	download_url="https://github.com/nvlabs/tiny-cuda-nn",
	license="BSD 3-Clause \"New\" or \"Revised\" License",
	packages=["tinycudann"],
	install_requires=[],
	include_package_data=True,
	zip_safe=False,
	ext_modules=[ext],
	cmdclass={"build_ext": BuildExtension},
)
