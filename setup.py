#!/usr/bin/env python3
# setup.py

from setuptools import setup
from setuptools.command.build import build as _build
import subprocess
import os
import sys
import shutil
import glob


class CustomBuild(_build):
    def run(self):
        orig_dir = os.getcwd()

        # Step 0: Cleanup
        paths = ["dakota-packages/OPTPP/build", "src/everest_optimizers/pyCONMIN/*.so"]
        for path in paths:
            for resolved_path in glob.glob(path):
                if os.path.isdir(resolved_path):
                    shutil.rmtree(resolved_path)
                elif os.path.isfile(resolved_path):
                    os.remove(resolved_path)

        # Step 1: Link trilinos
        optpp_dir = os.path.abspath("dakota-packages/OPTPP")
        trilinos_src = os.path.abspath("dakota-packages/trilinos")
        trilinos_link = os.path.join(optpp_dir, "trilinos")

        if not os.path.exists(trilinos_link):
            os.makedirs(optpp_dir, exist_ok=True)
            os.symlink(trilinos_src, trilinos_link)

        # Step 2: Get pybind11 CMake dir
        cmake_dir = subprocess.run(
            [sys.executable, "-m", "pybind11", "--cmakedir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Step 3: Run CMake configure
        build_dir = os.path.join(optpp_dir, "build")
        os.makedirs(build_dir, exist_ok=True)

        cmake_cmd = [
            "cmake",
            "-B",
            build_dir,
            "-S",
            optpp_dir,
            "-D",
            "CMAKE_BUILD_TYPE=Release",
            "-D",
            "DAKOTA_NO_FIND_TRILINOS=TRUE",
            "-D",
            "BUILD_SHARED_LIBS=ON",
            "-D",
            f"Python3_EXECUTABLE={sys.executable}",
            "-D",
            f"pybind11_DIR={cmake_dir}",
        ]

        subprocess.run(cmake_cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)

        # Step 4: CMake build
        subprocess.run(
            ["cmake", "--build", build_dir, "--", "-j"],
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Step 5: f2py
        f2py_dir = os.path.abspath("src/everest_optimizers/pyCONMIN")
        os.makedirs(f2py_dir, exist_ok=True)
        os.chdir(f2py_dir)

        f2py_cmd = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-m",
            "conmin",
            "-c",
            "source/f2py/conmin.pyf",
            "source/openunit.f",
            "source/cnmn00.f",
            "source/cnmn01.f",
            "source/cnmn02.f",
            "source/cnmn03.f",
            "source/cnmn04.f",
            "source/cnmn05.f",
            "source/cnmn06.f",
            "source/cnmn07.f",
            "source/cnmn08.f",
            "source/cnmn09.f",
            "source/conmin.f",
            "source/closeunit.f",
        ]
        subprocess.run(f2py_cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)

        os.chdir(orig_dir)
        super().run()


setup(
    cmdclass={"build": CustomBuild},
    zip_safe=False,
)
