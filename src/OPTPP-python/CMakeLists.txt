find_package(pybind11 REQUIRED)

pybind11_add_module(pyoptpp pyoptpp.cpp)

target_include_directories(pyoptpp PRIVATE ${optpp_SOURCE_DIR}/include)
target_link_libraries(pyoptpp PRIVATE optpp ${Teuchos_LIBRARIES})
install(TARGETS pyoptpp DESTINATION everest_optimizers)
