#!/usr/bin/env python3
# setup.py

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import subprocess
import os
import sys


class CustomBuildExt(build_ext):
    def run(self):
        # Step 0: Ensure dependencies
        print("Installing Python dependencies (pybind11, numpy)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pybind11", "numpy"],
            check=True, stdout=sys.stdout, stderr=sys.stderr
        )

        # Step 1: Link trilinos if needed
        optpp_dir = os.path.abspath("dakota-packages/OPTPP")
        trilinos_src = os.path.abspath("dakota-packages/trilinos")
        trilinos_link = os.path.join(optpp_dir, "trilinos")

        if not os.path.exists(trilinos_link):
            print(f"Creating symlink: {trilinos_link} -> {trilinos_src}")
            os.makedirs(optpp_dir, exist_ok=True)
            os.symlink(trilinos_src, trilinos_link)

        # Step 2: Get pybind11 cmake dir
        print("Querying pybind11 CMake directory...")
        cmake_dir = subprocess.run(
            [sys.executable, "-m", "pybind11", "--cmakedir"],
            check=True, capture_output=True, text=True
        ).stdout.strip()
        print(f"pybind11 CMake directory: {cmake_dir}")

        # Step 3: Run CMake configure
        build_dir = os.path.join(optpp_dir, "build")
        os.makedirs(build_dir, exist_ok=True)

        cmake_cmd = [
            "cmake", "-B", build_dir, "-S", optpp_dir,
            "-D", "CMAKE_BUILD_TYPE=Release",
            "-D", "DAKOTA_NO_FIND_TRILINOS=TRUE",
            "-D", "BUILD_SHARED_LIBS=ON",
            "-D", f"Python3_EXECUTABLE={sys.executable}",
            "-D", f"pybind11_DIR={cmake_dir}"
        ]

        print("Running CMake configuration...")
        subprocess.run(cmake_cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)

        # Step 4: Build with CMake
        print("Building project with CMake...")
        subprocess.run(
            ["cmake", "--build", build_dir, "--", "-j"],
            check=True, stdout=sys.stdout, stderr=sys.stderr
        )

        # Finally call regular build_ext
        print("Running standard build_ext...")
        super().run()


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
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
)
