# -*- coding: utf-8 -*-

import os
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NAME = 'mesh_intersection_cuda_bvh'
VERSION = '0.1.0'

here = os.path.abspath(os.path.dirname(__file__))

bvh_src_files = ['src/bvh.cpp', 'src/bvh_cuda_op.cu']
bvh_include_dirs = [
    'include',
    '/usr/mycpp/cuda/cu-116/cuda-samples/Common'
]

bvh_extra_compile_args = {
    'nvcc': ['-DPRINT_TIMINGS=0', '-DDEBUG_PRINT=0',
             '-DERROR_CHECKING=1', '-DCOLLISION_ORDERING=1'],
    'cxx': []
}

bvh_extension = CUDAExtension('bvh_cuda', bvh_src_files,
                              include_dirs=bvh_include_dirs,
                              extra_compile_args=bvh_extra_compile_args)

setup(
    name=NAME,
    version=VERSION,
    ext_modules=[bvh_extension],
    cmdclass={'build_ext': BuildExtension}
)
