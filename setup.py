#!/usr/bin/env python3
# setup.py

from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup
import os

# Self-contained OptQNewton implementation
ext_modules = [
    Pybind11Extension(
        "everest_optimizers",
        ["src/simple_test.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    name="everest-optimizers",
    version="1.0.0",
    author="Everest Optimizers",
    author_email="",
    url="https://github.com/equinor/everest-optimizers",
    description="Everest optimization algorithms for Python",
    long_description="""
    Everest optimization algorithms package providing high-performance optimization tools.
    
    This package provides Python access to advanced optimization algorithms including
    OptQNewton quasi-Newton optimization and related functionality, designed for
    integration into Python workflows.
    
    Features:
    - Quasi-Newton optimization with BFGS Hessian approximation
    - Support for different globalization strategies (line search, trust region)
    - NumPy integration for easy data exchange
    - Simple Python interface for optimization problems
    """,
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.15",
    ],
    zip_safe=False,
    cmdclass={"build_ext": build_ext},
)