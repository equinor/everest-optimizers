# make sure that Eigen supports the "Ref" command
set(CMAKE_REQUIRED_INCLUDES ${STANMATH_INCLUDE_DIR})
set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS}")

CHECK_CXX_SOURCE_COMPILES(
  "
  #include <stan/math/fwd/scal.hpp>
  int main(){
    return 0;
  }
  "
  STANMATH_INCLUDES_EXIST)


if(NOT STANMATH_INCLUDES_EXIST)
	set(STANMATH_TEST_FAIL 1)
	set(MUQ_USE_STANMATH OFF)
else()
	set(STANMATH_TEST_FAIL 0)
	set(MUQ_USE_STANMATH ON)
endif()




