find_package(pybind11 REQUIRED)

include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281
) # 10.2.1
FetchContent_MakeAvailable(fmt)

pybind11_add_module(pyoptpp pyoptpp.cpp)

target_include_directories(pyoptpp PRIVATE ${optpp_SOURCE_DIR}/include)
target_link_libraries(pyoptpp PRIVATE optpp ${Teuchos_LIBRARIES} fmt::fmt)
install(TARGETS pyoptpp DESTINATION everest_optimizers)
