from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import platform

# 检测操作系统类型
is_mac = platform.system() == "Darwin"
is_linux = platform.system() == "Linux"
is_windows = platform.system() == "Windows"

# 设置编译器标志
compile_args = []
link_args = []

if is_mac:
    compile_args += [
        "-std=c++14",
        "-O3",
        "-Wall",
        "-Wextra",
        "-fPIC",
        "-Wno-unused-function",
        "-Wno-unused-parameter",
        "-stdlib=libc++",
        "-mmacosx-version-min=10.14",
    ]
    link_args += ["-stdlib=libc++", "-mmacosx-version-min=10.14"]
elif is_linux:
    compile_args += ["-std=c++14", "-O3", "-Wall", "-Wextra", "-fPIC"]
elif is_windows:
    compile_args += ["/O2", "/Wall", "/std:c++14"]


# 定义pybind11的路径
class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


# 定义扩展模块
ext_modules = [
    Extension(
        "mesh_graph_cut_cpp",
        [
            "mesh_graph_cut/Cpp/src/bindings.cpp",
            "mesh_graph_cut/Cpp/src/region_growing.cpp",
            "mesh_graph_cut/Cpp/src/kdtree.cpp",
            "mesh_graph_cut/Cpp/src/halfedge.cpp",
        ],
        include_dirs=[
            # 包含目录
            "mesh_graph_cut/Cpp/include",
            get_pybind_include(),
            get_pybind_include(user=True),
            # 添加Eigen库的包含路径
            "/usr/local/include/eigen3",
            "/usr/include/eigen3",
        ],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++",
    ),
]


# 自定义构建扩展命令
class BuildExt(build_ext):
    def build_extensions(self):
        # 检查编译器是否支持C++14
        if is_windows:
            if self.compiler.compiler_type == "msvc":
                # 使用MSVC编译器
                for ext in self.extensions:
                    ext.extra_compile_args = ["/O2", "/Wall", "/std:c++14"]
        else:
            # 使用GCC或Clang编译器
            try:
                self.compiler.compiler_so.remove("-Wstrict-prototypes")
            except (AttributeError, ValueError):
                pass
        build_ext.build_extensions(self)


setup(
    name="mesh_graph_cut_cpp",
    version="0.1.0",
    author="chli",
    author_email="chli@example.com",
    description="A fast mesh graph cutting library with C++ backend",
    long_description="",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "open3d>=0.13.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.48.0",
        "joblib>=0.16.0",
        "tqdm-joblib>=0.0.1",
        "pybind11>=2.6.0",
    ],
    zip_safe=False,
)
