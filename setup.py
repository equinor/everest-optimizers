#!/usr/bin/env python3
# setup.py
"""Setup for everest optimizers."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Extensions
ext_modules = [
    Pybind11Extension(
        "everest_optimizers",
        ["src/simple_test.cpp"],
        language="c++",
        cxx_std=17,
    ),
    Pybind11Extension(
        "everest_optimizers_test",
        ["src/OPTQNewton.cpp"],
        language="c++",
        cxx_std=17,
        include_dirs=[
      "include",
      "trilinos/packages/teuchos/core/src",
      "trilinos/packages/teuchos/core/cmake",
      "trilinos/packages/teuchos/numerics/src",
      "trilinos/packages/teuchos/comm/src",
  ]
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
