# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from setuptools import setup
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

code_dir = os.path.dirname(os.path.realpath(__file__))

# 设置C++和CUDA编译器标志
nvcc_flags = [
    '-Xcompiler', '-O3', '-std=c++17',  # 使用C++17标准
    '-U__CUDA_NO_HALF_OPERATORS__', 
    '-U__CUDA_NO_HALF_CONVERSIONS__', 
    '-U__CUDA_NO_HALF2_OPERATORS__'
]
c_flags = ['-O3', '-std=c++17']  # 使用C++17标准

# 定义扩展模块
ext_modules = [
    CUDAExtension(
        name='common',
        sources=[
            'bindings.cpp',
            'common.cu',
        ],
        extra_compile_args={
            'gcc': c_flags,
            'nvcc': nvcc_flags
        }
    ),
    CUDAExtension(
        name='gridencoder',
        sources=[
            f"{code_dir}/torch_ngp_grid_encoder/gridencoder.cu",
            f"{code_dir}/torch_ngp_grid_encoder/bindings.cpp",
        ],
        extra_compile_args={
            'gcc': c_flags,
            'nvcc': nvcc_flags
        }
    )
]

# 设置安装包
setup(
    name='common',
    ext_modules=ext_modules,
    include_dirs=[
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)