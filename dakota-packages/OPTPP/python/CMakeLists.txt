# /home/svein/equinor/temp/everest-optimizers/dakota-packages/OPTPP/python/CMakeLists.txt

find_package(pybind11 REQUIRED)

pybind11_add_module(pyopttpp
  pyopttpp.cpp
  ../tests/constraint_tests.C
)

target_include_directories(pyopttpp PRIVATE
  ../tests
  ../include
)

target_link_libraries(pyopttpp PRIVATE
  optpp
  ${Teuchos_LIBRARIES}
)

# Ensure the module is placed in a predictable location within the build directory
set_target_properties(pyopttpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

install(TARGETS pyopttpp DESTINATION lib/python)
