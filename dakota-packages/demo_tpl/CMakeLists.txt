cmake_minimum_required(VERSION 3.17)
project("DEMO_TPL" CXX)

set(CMAKE_CXX_STANDARD 11 CACHE STRING
  "DEMO_TPL strictly requires C++11 or better")

SUBDIRS(src)

if( DEMO_TPL_DRIVER )
  add_executable(driver src/main.cpp)
  target_link_libraries( driver demo_tpl )
endif()
