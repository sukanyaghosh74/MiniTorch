from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'minitorch._C',
        ['cpp/src/bindings.cpp', 'cpp/src/tensor.cpp'],
        include_dirs=[
            'cpp/include',
            get_pybind_include(),
        ],
        language='c++',
    ),
]

setup(
    name='minitorch',
    version='0.1.0',
    author='MiniTorch Contributors',
    description='A compact PyTorch-like ML framework',
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    install_requires=['numpy', 'pybind11'],
    python_requires='>=3.7',
    zip_safe=False,
)
