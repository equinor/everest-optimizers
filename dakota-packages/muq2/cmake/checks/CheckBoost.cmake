# make sure that the boost graph library is available
set(CMAKE_REQUIRED_LIBRARIES ${BOOST_LIBRARY})
set(CMAKE_REQUIRED_INCLUDES ${BOOST_INCLUDE_DIR})
set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS}")
CHECK_CXX_SOURCE_COMPILES(
"
#include <boost/graph/adjacency_list.hpp>
typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::bidirectionalS, int, int> Graph;
int main(){
Graph temp;
return 0;
}
"
BOOST_GRAPH_COMPILES)


if(NOT BOOST_GRAPH_COMPILES)
	set(BOOST_TEST_FAIL 1)
else()
	set(BOOST_TEST_FAIL 0)
endif()
