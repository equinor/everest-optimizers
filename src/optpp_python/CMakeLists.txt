find_package(pybind11 REQUIRED)

pybind11_add_module(_pyoptpp pyoptpp.cpp)

target_compile_definitions(_pyoptpp PRIVATE DAKOTA_OPTPP)
target_include_directories(_pyoptpp PRIVATE ${optpp_SOURCE_DIR}/include)
target_link_libraries(_pyoptpp PRIVATE optpp ${Teuchos_LIBRARIES})
install(TARGETS _pyoptpp DESTINATION everest_optimizers)
