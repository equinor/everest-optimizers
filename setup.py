#!/usr/bin/env python3
# setup.py

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Self-contained OptQNewton implementation
ext_modules = [
    Pybind11Extension(
        "everest_optimizers",
        ["src/simple_test.cpp"],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
