cmake_minimum_required(VERSION 3.23.0 FATAL_ERROR)
project(everest-optimizers)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter NumPy)

if(DEBUG_MEMORY)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
endif()

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.1
    FIND_PACKAGE_ARGS NAMES pybind11
)
FetchContent_MakeAvailable(pybind11)

# Build OPTPP. trilinos is also needed to build it.
message(STATUS "Downloading OPTPP")
FetchContent_Declare(
    optpp_packages
    GIT_REPOSITORY https://github.com/snl-dakota/dakota-packages
    GIT_TAG e83c52d7519d2421dffdfbd2b6fc6f7e54a86727
    SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/optpp_packages"
    PATCH_COMMAND
        git apply
            "${CMAKE_CURRENT_SOURCE_DIR}/OPTPP-van-shanno-fix.patch"
            "${CMAKE_CURRENT_SOURCE_DIR}/OPTPP-null-default-output.patch"
            "${CMAKE_CURRENT_SOURCE_DIR}/OPTPP-decimal-output.patch"
    # exclude in case the repository adds a root CMakeLists.txt file
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(optpp_packages)
message(STATUS "Finished downloading OPTPP")

set(Teuchos_DIR ${CMAKE_CURRENT_BINARY_DIR}/trilinos/packages/teuchos)
set(Trilinos_ENABLE_Teuchos ON CACHE BOOL "" FORCE)
add_subdirectory(${optpp_packages_SOURCE_DIR}/trilinos EXCLUDE_FROM_ALL)
set(DAKOTA_OPTPP ON CACHE BOOL "Build the Dakota OPT++ interface" FORCE)
add_subdirectory(${optpp_packages_SOURCE_DIR}/OPTPP EXCLUDE_FROM_ALL)
add_subdirectory(src/optpp_python)

# Install our python files
file(GLOB EVEREST_OPTIMIZERS_SRC src/everest_optimizers/*.py)
install(FILES ${EVEREST_OPTIMIZERS_SRC} DESTINATION everest_optimizers)
