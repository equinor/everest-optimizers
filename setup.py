#!/usr/bin/env python3
# setup.py

from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup
import os

# Self-contained OptQNewton implementation
ext_modules = [
    Pybind11Extension(
        "optpp_bindings",
        ["src/simple_test.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    name="optpp_bindings",
    version="1.0.0",
    author="OPTPP Python Bindings",
    author_email="",
    url="https://github.com/your-repo/optpp-python-bindings",
    description="Python bindings for OPTPP optimization library",
    long_description="""
    Python bindings for the OPTPP (OPTimization++) library from Sandia National Laboratories.
    
    This package provides Python access to the OptQNewton quasi-Newton optimization algorithm
    and related functionality from OPTPP, allowing easy integration of high-performance
    optimization algorithms into Python workflows.
    
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