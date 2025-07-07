include(ExternalProject)
if(NOT OTF2_EXTERNAL_SOURCE)
	set(OTF2_EXTERNAL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/external/otf2-2.2.tar.gz)
endif()

set(OTF2_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/muq_external)

set(OTF2_CFLAGS "")
if(CMAKE_OSX_ARCHITECTURES)
  set(OTF2_CFLAGS "${OTF2_CFLAGS}-arch arm64 -arch x86_64")
endif()


ExternalProject_Add(
	OTF2
                PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/otf2
                URL ${OTF2_EXTERNAL_SOURCE}
                BUILD_IN_SOURCE TRUE
                CONFIGURE_COMMAND ./configure --prefix=${OTF2_INSTALL_DIR} CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} CFLAGS=${OTF2_CFLAGS} CXXFLAGS=${OTF2_CFLAGS}
                BUILD_COMMAND $(MAKE)
                INSTALL_COMMAND make install
)

set(OTF2_LIBRARIES ${OTF2_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}otf2${CMAKE_STATIC_LIBRARY_SUFFIX})
set(OTF2_LIBRARY ${OTF2_LIBRARIES})

set(OTF2_INCLUDE_DIRS ${OTF2_INSTALL_DIR}/include)
message(STATUS "Adding ${OTF2_INSTALL_DIR} for an OTF2 include directory.")

set(MUQ_HAS_OTF2 1)
