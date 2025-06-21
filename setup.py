import os
import glob
import torch
from platform import system
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension

SYSTEM = system()

cut_root_path = os.getcwd() + "/mesh_graph_cut/Cpp/"
cut_lib_path = os.getcwd() + "/mesh_graph_cut/Lib/"
cut_src_path = cut_root_path + "src/"
cut_sources = glob.glob(cut_src_path + "*.cpp")
cut_include_dirs = [
    cut_root_path + "include",
    cut_lib_path + "mcut/include",
]

cut_library_dirs = [os.path.abspath(cut_lib_path + "mcut/build/bin")]
cut_libraries = ["mcut"]

cut_extra_compile_args = [
    "-O3",
    "-Wall",
    "-Wextra",
    "-fPIC",
    "-DCMAKE_BUILD_TYPE=Release",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
    "-DTORCH_USE_CUDA_DSA",
]

link_args = []

if SYSTEM == "Darwin":
    cut_extra_compile_args.append("-std=c++17")
    cut_extra_compile_args += [
        "-Wno-unused-function",
        "-Wno-unused-parameter",
        "-stdlib=libc++",
        "-mmacosx-version-min=10.14",
        "-fopenmp",
    ]
    link_args += [
        "-stdlib=libc++",
        "-mmacosx-version-min=10.14",
        "-lomp",  # macOS需要链接libomp
    ]
elif SYSTEM == "Linux":
    cut_extra_compile_args.append("-std=c++17")
    cut_extra_compile_args += [
        "-fopenmp",
    ]
    link_args += ["-fopenmp"]

if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    arch_str = f"{cc[0]}.{cc[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_str

    cut_sources += glob.glob(cut_src_path + "*.cu")

    extra_compile_args = {
        "cxx": cut_extra_compile_args
        + [
            "-DUSE_CUDA",
            "-DTORCH_USE_CUDA_DSA",
        ],
        "nvcc": [
            "-O3",
            "-Xfatbin",
            "-compress-all",
            "-DUSE_CUDA",
            "-std=c++17",
            "-DTORCH_USE_CUDA_DSA",
        ],
    }

    cut_module = CUDAExtension(
        name="cut_cpp",
        sources=cut_sources,
        include_dirs=cut_include_dirs,
        library_dirs=cut_library_dirs,
        libraries=cut_libraries,
        runtime_library_dirs=cut_library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=link_args,
    )

else:
    cut_module = CppExtension(
        name="cut_cpp",
        sources=cut_sources,
        include_dirs=cut_include_dirs,
        library_dirs=cut_library_dirs,
        libraries=cut_libraries,
        runtime_library_dirs=cut_library_dirs,
        extra_compile_args=cut_extra_compile_args,
        extra_link_args=link_args,
    )

setup(
    name="CUT-CPP",
    version="1.0.0",
    author="Changhao Li",
    packages=find_packages(),
    ext_modules=[cut_module],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
