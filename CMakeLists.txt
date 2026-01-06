cmake_minimum_required(VERSION 3.23.0 FATAL_ERROR)
project(everest-optimizers)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter NumPy)

# Build OPTPP. trilinos is also needed to build it.
include(FetchContent)
message(STATUS "Downloading dakota-packages")
FetchContent_Declare(
    dakota-packages
    GIT_REPOSITORY https://github.com/snl-dakota/dakota-packages
    GIT_TAG e83c52d7519d2421dffdfbd2b6fc6f7e54a86727
    SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/third-party/dakota-packages"
    PATCH_COMMAND
        git apply "${CMAKE_CURRENT_SOURCE_DIR}/OPTPP-van-shanno-fix.patch"
    # exclude in case dakota-packages adds a root CMakeLists.txt file
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(dakota-packages)
message(STATUS "Finished downloading dakota-packages")

set(Teuchos_DIR ${CMAKE_CURRENT_BINARY_DIR}/trilinos/packages/teuchos)
set(Trilinos_ENABLE_Teuchos ON CACHE BOOL "" FORCE)
add_subdirectory(${dakota-packages_SOURCE_DIR}/trilinos EXCLUDE_FROM_ALL)
add_subdirectory(${dakota-packages_SOURCE_DIR}/OPTPP EXCLUDE_FROM_ALL)
add_subdirectory(src/OPTPP-python)

# Build pyoptsparse for CONMIN support
include(ExternalProject)
set(MOD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/third-party/pyoptsparse")
set(MOD_TARGET "conmin.${Python_SOABI}.so")
message(STATUS "Downloading and configuring pyoptsparse")
ExternalProject_Add(
    pyCONMIN
    GIT_REPOSITORY https://github.com/mdolab/pyoptsparse
    GIT_TAG
        85549852b70d757de0e3acdcaa6c1fdf5458ce53 # v2.14.3
    SOURCE_DIR ${MOD_SRC}
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/_pyCONMIN"
    CONFIGURE_COMMAND meson setup . ${MOD_SRC} --buildtype=release
    BUILD_COMMAND meson compile ${MOD_TARGET}
    # meson install currently does nothing. That will change if
    # https://github.com/mdolab/pyoptsparse/pull/433 is merged.
    INSTALL_COMMAND meson install --destdir ${CMAKE_BINARY_DIR}
    # patching out all the modules we did not compile from pyoptsparse/__init__.py
    PATCH_COMMAND
        patch pyoptsparse/__init__.py ${CMAKE_SOURCE_DIR}/pyoptsparse.patch
)
message(STATUS "Finished configuring pyoptsparse")

install(
    FILES
        "${CMAKE_BINARY_DIR}/python/_pyCONMIN/pyoptsparse/pyCONMIN/${MOD_TARGET}"
        third-party/pyoptsparse/pyoptsparse/pyCONMIN/__init__.py
        third-party/pyoptsparse/pyoptsparse/pyCONMIN/pyCONMIN.py
        third-party/pyoptsparse/pyoptsparse/pyCONMIN/LICENSE
    DESTINATION everest_optimizers/pyoptsparse/pyCONMIN
)
install(
    FILES
        third-party/pyoptsparse/LICENSE
        third-party/pyoptsparse/pyoptsparse/__init__.py
        third-party/pyoptsparse/pyoptsparse/pyOpt_MPI.py
        third-party/pyoptsparse/pyoptsparse/pyOpt_constraint.py
        third-party/pyoptsparse/pyoptsparse/pyOpt_error.py
        third-party/pyoptsparse/pyoptsparse/pyOpt_gradient.py
        third-party/pyoptsparse/pyoptsparse/pyOpt_history.py
        third-party/pyoptsparse/pyoptsparse/pyOpt_objective.py
        third-party/pyoptsparse/pyoptsparse/pyOpt_optimization.py
        third-party/pyoptsparse/pyoptsparse/pyOpt_optimizer.py
        third-party/pyoptsparse/pyoptsparse/pyOpt_solution.py
        third-party/pyoptsparse/pyoptsparse/pyOpt_types.py
        third-party/pyoptsparse/pyoptsparse/pyOpt_utils.py
        third-party/pyoptsparse/pyoptsparse/pyOpt_variable.py
    DESTINATION everest_optimizers/pyoptsparse
)

# Install our python files
file(GLOB EVEREST_OPTIMIZERS_SRC src/everest_optimizers/*.py)
install(FILES ${EVEREST_OPTIMIZERS_SRC} DESTINATION everest_optimizers)
