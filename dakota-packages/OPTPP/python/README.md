# `pyopttpp`: Python Wrapper for OPTPP

This directory contains the `pybind11` wrapper for the OPTPP C++ library, allowing you to use the `OptQNewton` optimizer from Python.

## Prerequisites

Before building, ensure you have the following installed:
- A C++ compiler (e.g., `g++`)
- `cmake` (version 3.17 or newer)
- `make`
- `python` (3.6+) and `pip`
- `git`

## Setup and Build

Follow these steps to set up the environment and build the Python module.

### Step 1: Install Python Dependencies

Install `pybind11` and `numpy` using pip. It is recommended to use a Python virtual environment.

```bash
pip install pybind11 numpy
```

### Step 2: Set up Trilinos Dependency

The build system expects the `trilinos` source directory to be present inside the `OPTPP` directory. If your `trilinos` checkout is located elsewhere (e.g., in `dakota-packages/trilinos`), create a symbolic link from within the `OPTPP` directory:

```bash
# From within the dakota-packages/OPTPP directory
ln -sfn ../trilinos trilinos
```

### Step 3: Enable the Python Wrapper Build

Before configuring the project, you must enable the Python wrapper build in the main `CMakeLists.txt` file.

1.  Open `/path/to/dakota-packages/OPTPP/CMakeLists.txt`.
2.  Find the line `add_subdirectory(src)`.
3.  Add the following line immediately after it:
    ```cmake
    add_subdirectory(python)
    ```

This ensures that the build system will compile the `pyopttpp` module.

### Step 4: Configure with CMake and Compile

These commands will configure the project with CMake and then compile the C++ library and the Python wrapper.

```bash
# Navigate to the OPTPP root directory
cd /path/to/dakota-packages/OPTPP

# Create a build directory and configure the project
cmake -B build -S . \
      -D CMAKE_BUILD_TYPE=Release \
      -D DAKOTA_NO_FIND_TRILINOS=TRUE \
      -D BUILD_SHARED_LIBS=ON

# Compile the project
cmake --build build -- -j
```
This will create the Python module file (`pyopttpp.cpython-*.so`) in the `build/python` directory.

**Troubleshooting:**
- **`pybind11` not found:** If CMake cannot find `pybind11`, you must provide the path manually. You can find the path by running `python -m pybind11 --cmakedir`. Then, add the following flag to the `cmake` command, replacing the path with the output from the previous command:
  ```
  -D pybind11_DIR=/path/to/pybind11/share/cmake/pybind11
  ```
- **Wrong Python interpreter:** If CMake picks up the wrong Python interpreter, you can specify it with `-D Python3_EXECUTABLE=$(which python)`.

### Step 5: Run the Test Script

To verify that the module was built correctly, run the provided test script from the `python` directory.

```bash
# Navigate to the python directory
cd /path/to/dakota-packages/OPTPP/python

# Run the test script with the Python interpreter used for the build
python test_opttpp.py
```

You should see the optimizer converge and a "Test Passed!" message.

## Note on Memory Leaks

If you run the test script with a memory checker like AddressSanitizer (e.g., using `LD_PRELOAD`), it may report memory leaks upon interpreter shutdown. This is expected behavior.

The underlying C++ libraries (`OPTPP` and `Teuchos`) allocate static global resources that are not deallocated before the Python interpreter exits. The `pybind11` documentation acknowledges this as a common issue when wrapping such libraries and recommends allowing the operating system to reclaim this memory when the process terminates. These reported "leaks" are a one-time allocation and do not indicate a runtime memory issue.
