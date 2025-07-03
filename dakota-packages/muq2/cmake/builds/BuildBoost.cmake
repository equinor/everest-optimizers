include(ExternalProject)

if(NOT DEFINED BOOST_EXTERNAL_SOURCE)
  set(BOOST_EXTERNAL_SOURCE http://downloads.sourceforge.net/project/boost/boost/1.78.0/boost_1_78_0.tar.gz)
endif()



set(BOOST_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/boost/src/BOOST")

# decide what toolset boost should use, start with compiler types, then work through operating systems
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")

  # using Intel C++
  if(WIN32)
    set(BOOST_TOOLSET_NAME intel-win32)
  else()
    set(BOOST_TOOLSET_NAME intel-linux)
  endif()

  set(BOOST_CXX_FLAGS "")
  set(BOOST_LINK_FLAGS "")

# is this an OSX machine?
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

  set(BOOST_TOOLSET_NAME clang)

  if(MUQ_USE_LIBC11)
  	set(BOOST_CXX_FLAGS "-stdlib=libc++")
  	set(BOOST_LINK_FLAGS "-stdlib=libc++")
  else(MUQ_USE_LIBC11)
    set(BOOST_CXX_FLAGS "")
    set(BOOST_LINK_FLAGS "")
  endif(MUQ_USE_LIBC11)

# is the compiler clang?
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")

    set(BOOST_TOOLSET_NAME clang)

    if(MUQ_USE_LIBC11)
    	set(BOOST_CXX_FLAGS "-stdlib=libc++")
    	set(BOOST_LINK_FLAGS "-stdlib=libc++")
    else(MUQ_USE_LIBC11)
        set(BOOST_CXX_FLAGS "")
        set(BOOST_LINK_FLAGS "")
    endif(MUQ_USE_LIBC11)

# is the compiler g++?
elseif(CMAKE_COMPILER_IS_GNUCXX)

    set(BOOST_TOOLSET_NAME gcc)

    set(BOOST_CXX_FLAGS "")
    set(BOOST_LINK_FLAGS "")

# is this an windows machine?
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

  if(MINGW)
    #using MINGW
    set(BOOST_TOOLSET_NAME mingw)
  elseif(MSYS)
    # using Visual Studio C++
    set(BOOST_TOOLSET_NAME msvc)
  else()
    message( FATAL_ERROR "Unable to find a BOOST toolset that matches your compiler and system.  Either use a different compiler, or try installing boost manually." )
  endif()

  set(BOOST_CXX_FLAGS "")
  set(BOOST_LINK_FLAGS "")

else()
        message( FATAL_ERROR "Unable to find a BOOST toolset that matches your compiler and system.  Either use a different compiler, or try installing boost manually." )
endif()


set(Boost_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/muq_external)

# create a configure file for boost_1_55_0
message(STATUS "Creating ${BOOST_BUILD_DIR}/tools/build/v2/user-config.jam")
string(REGEX MATCH "[0-9]+\\.[0-9]+" BOOST_TOOLSET_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
if(MUQ_USE_OPENMPI)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/builds/user-config-mpi.jam.in ${CMAKE_CURRENT_BINARY_DIR}/user-config.jam)
else()
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/builds/user-config.jam.in ${CMAKE_CURRENT_BINARY_DIR}/user-config.jam)
endif()

message(STATUS "BOOST_LINK_FLAGS = ${BOOST_LINK_FLAGS}")
message(STATUS "BOOST_CXX_FLAGS = ${BOOST_CXX_FLAGS}")


# Do we want to compile for multiple architectures on OSX (on an M1 for example)?
# If so, we need to do some trickery
if(CMAKE_OSX_ARCHITECTURES)

  message(STATUS "BUILDING BOOST WITH OSX ARCHITECTURES")
  # ./bootstrap.sh --with-toolset=clang-darwin cxxflags="-arch x86_64 -arch arm64" cflags="-arch x86_64 -arch arm64" linkflags="-arch x86_64 -arch arm64" --with-libraries=graph,system,filesystem
  # ./b2 toolset=clang-darwin target-os=darwin architecture=x86 cxxflags="-arch x86_64" cflags="-arch x86_64" linkflags="-arch x86_64" stage
  # mkdir x86
  # mv stage/lib/*.dylib x86
  # ./b2 toolset=clang-darwin target-os=darwin architecture=arm abi=aapcs cxxflags="-arch arm64" cflags="-arch arm64" linkflags="-arch arm64" stage
  # mkdir arm
  # mv stage/lib/*.dylib arm
  # for dylib in arm*; do
  #   lipo -create -arch arm64 $dylib -arch x86_64 x86/$(basename $dylib) -output stage/lib/$(basename $dylib);
  # done
# test ./b2 toolset=clang-darwin target-os=darwin architecture=x86 cxxflags="-arch x86_64" cflags="-arch x86_64" linkflags="-arch x86_64" stage && cp -r stage/lib/ x86-libs/ && ./b2 toolset=clang-darwin target-os=darwin architecture=arm abi=aapcs cxxflags="-arch arm64" cflags="-arch arm64" linkflags="-arch arm64" stage && cp -r stage/lib arm-libs/ && for dylib in arm-libs/*; do lipo -create -arch arm64 $dylib -arch x86_64 x86/$(basename $dylib) -output stage/lib/$(basename $dylib); done

  ExternalProject_Add(
    BOOST
    BUILD_IN_SOURCE 1
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/boost
    URL ${BOOST_EXTERNAL_SOURCE}
    PATCH_COMMAND ""
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ./bootstrap.sh --prefix=${Boost_INSTALL_DIR} --with-toolset=clang cxxflags="-arch x86_64 -arch arm64" cflags="-arch x86_64 -arch arm64" linkflags="-arch x86_64 -arch arm64" --with-libraries=graph
    BUILD_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/builds/boost-build-universal-osx.sh ${Boost_INSTALL_DIR}
    INSTALL_COMMAND ""
    )


else(CMAKE_OSX_ARCHITECTURES)

  if(MUQ_USE_LIBC11)
      ExternalProject_Add(
          BOOST
          PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/boost
          URL ${BOOST_EXTERNAL_SOURCE}
          PATCH_COMMAND ""
          UPDATE_COMMAND ""
          CONFIGURE_COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/user-config.jam ${BOOST_BUILD_DIR}/user-config.jam && ${BOOST_BUILD_DIR}/bootstrap.sh --prefix=${Boost_INSTALL_DIR} --without-icu
          BUILD_COMMAND ${BOOST_BUILD_DIR}/b2 cxxflags=${BOOST_CXX_FLAGS} linkflags=${BOOST_LINK_FLAGS} variant=release --user-config=${BOOST_BUILD_DIR}/user-config.jam toolset=${BOOST_TOOLSET_NAME}-muq --with-graph--disable-icu install
          BUILD_IN_SOURCE 1
          INSTALL_COMMAND ""
          )
  else(MUQ_USE_LIBC11)
          ExternalProject_Add(
            BOOST
            PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/boost
            URL ${BOOST_EXTERNAL_SOURCE}
            PATCH_COMMAND ""
            UPDATE_COMMAND ""
            CONFIGURE_COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/user-config.jam ${BOOST_BUILD_DIR}/user-config.jam && ${BOOST_BUILD_DIR}/bootstrap.sh --prefix=${Boost_INSTALL_DIR} --without-icu
            BUILD_COMMAND ${BOOST_BUILD_DIR}/b2 cxxflags=${BOOST_CXX_FLAGS} variant=release --user-config=${BOOST_BUILD_DIR}/user-config.jam toolset=${BOOST_TOOLSET_NAME}-muq --with-graph --disable-icu  install
            BUILD_IN_SOURCE 1
            INSTALL_COMMAND ""
            )
   endif(MUQ_USE_LIBC11)
endif(CMAKE_OSX_ARCHITECTURES)

set_property( TARGET BOOST PROPERTY FOLDER "Externals")

set( Boost_INCLUDE_DIRS ${Boost_INSTALL_DIR}/include )

#list(APPEND Boost_LIBRARIES ${Boost_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}boost_filesystem${CMAKE_SHARED_LIBRARY_SUFFIX})
#list(APPEND Boost_LIBRARIES ${Boost_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}boost_system${CMAKE_SHARED_LIBRARY_SUFFIX})
list(APPEND Boost_LIBRARIES ${Boost_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}boost_graph${CMAKE_SHARED_LIBRARY_SUFFIX})
list(APPEND Boost_LIBRARIES ${Boost_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}boost_regex${CMAKE_SHARED_LIBRARY_SUFFIX})

#list(APPEND Boost_LIBRARIES_STATIC ${Boost_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}boost_filesystem${CMAKE_STATIC_LIBRARY_SUFFIX})
#list(APPEND Boost_LIBRARIES_STATIC ${Boost_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}boost_system${CMAKE_STATIC_LIBRARY_SUFFIX})
list(APPEND Boost_LIBRARIES_STATIC ${Boost_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}boost_graph${CMAKE_STATIC_LIBRARY_SUFFIX})
list(APPEND Boost_LIBRARIES_STATIC ${Boost_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}boost_regex${CMAKE_STATIC_LIBRARY_SUFFIX})
