import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')

extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '--std=c++17',  
        '-gencode=arch=compute_86,code=sm_86',
    ]
}

setup(
    name='llama_pure_compute',
    version='0.1.0',
    description='High-Performance C++/CUDA & Triton LLM Inference Engine',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    ext_modules=[
        CUDAExtension(
            name='llama_pure_compute._C',
            sources=[
                'src/csrc/ops/bindings.cpp',
                'src/csrc/ops/rmsswiglu.cu',
                'src/csrc/ops/rope.cu',
                'src/csrc/ops/kv_cache.cu',
            ],
            include_dirs=[
                os.path.abspath('src/csrc/include'),
                os.path.abspath('src/csrc/ops'),  
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=[f'-Wl,-rpath,{torch_lib_dir}'],  # Fixes libc10.so automatically
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0.0',
        'triton>=2.0.0',
        'pytest',
    ],
)