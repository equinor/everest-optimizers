#!/usr/bin/env python3
# setup.py
"""Setup for everest optimizers."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from setuptools.command.test import test as TestCommand
import subprocess
import sys
import os

class CustomTestCommand(TestCommand):
    def run_tests(self):
        # Step 1: Create symlink
        print("Linking trilinos...")
        subprocess.run(["ln", "-sfn", "../trilinos", "trilinos"], cwd="dakota-packages/OPTPP", check=True)

        # Step 2: Get pybind11 cmake directory
        print("Finding pybind11 cmake dir...")
        result = subprocess.run(["python3", "-m", "pybind11", "--cmakedir"], capture_output=True, text=True, check=True)
        pybind11_dir = result.stdout.strip()

        # Step 3: Configure with CMake
        print("Configuring project with CMake...")
        build_dir = os.path.abspath("dakota-packages/OPTPP/build")
        source_dir = os.path.abspath("dakota-packages/OPTPP")

        subprocess.run([
            "cmake", "-B", build_dir, "-S", source_dir,
            "-D", "CMAKE_BUILD_TYPE=Release",
            "-D", "DAKOTA_NO_FIND_TRILINOS=TRUE",
            "-D", "BUILD_SHARED_LIBS=ON",
            "-D", f"Python3_EXECUTABLE={sys.executable}",
            "-D", f"pybind11_DIR={pybind11_dir}"
        ], check=True)

        # Step 4: Compile
        print("Building project...")
        subprocess.run(["cmake", "--build", build_dir, "--", "-j"], check=True)

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
    cmdclass={
    "build_ext": build_ext,
    "custom_build": CustomTestCommand
},
    zip_safe=False,
)
