#!/bin/bash

# Build x86_64 libraries
./b2 toolset=clang target-os=darwin architecture=x86 cxxflags="-arch x86_64" cflags="-arch x86_64" linkflags="-arch x86_64" stage
mkdir -p x86-libs
ls stage/lib/*.dylib
mv stage/lib/*.dylib x86-libs

# Build arm libraries
./b2 toolset=clang target-os=darwin architecture=arm abi=aapcs cxxflags="-arch arm64" cflags="-arch arm64" linkflags="-arch arm64" stage
mkdir -p arm-libs
mv stage/lib/*.dylib arm-libs
ls arm-libs

# Combine the libraries to create a universal library
for dylib in arm-libs/*; do
  lipo -create -arch arm64 $dylib -arch x86_64 x86-libs/$(basename $dylib) -output stage/lib/$(basename $dylib);
done

# "Install" the headers and universal libraries
mkdir -p $1/include
cp -r boost $1/include
mkdir -p $1/lib
cp -r stage/lib/*  $1/lib
