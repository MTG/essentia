#!/usr/bin/env bash
set -e
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

# Prerequisites:        python>=3.10
#                       cmake>=3.28

echo "Building onnxruntime $LIBONNXRUNTIME_VERSION"

curl -SLO "https://github.com/microsoft/onnxruntime/archive/refs/tags/v$LIBONNXRUNTIME_VERSION.tar.gz"

tar -xf v$LIBONNXRUNTIME_VERSION.tar.gz
cd onnxruntime-$LIBONNXRUNTIME_VERSION

python3 -m pip install cmake

# Build the dynamic library for Linux
# ./build.sh \
#   --config RelWithDebInfo \
#   --build_shared_lib \
#   --parallel \
#   --compile_no_warning_as_error \
#   --skip_submodule_sync

# Build the dynamic library for MacOS (build for Intel and Apple silicon CPUs --> "x86_64;arm64")
./build.sh \
  --config RelWithDebInfo \
  --build_shared_lib \
  --parallel \
  --compile_no_warning_as_error \
  --skip_submodule_sync \
  --cmake_extra_defines CMAKE_OSX_ARCHITECTURES="arm64" FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER CMAKE_INSTALL_PREFIX=$PREFIX

#! We have found some issues building for cross-platforms, it looks it is much better to build it in a docker
#! In MacOS, we have experienced issues with the brew package. So, it needs to uninstall brew applications first (brew unsnstall onnxruntime)

# copying .pc file
mkdir -p "${PREFIX}"/lib/pkgconfig/
cp build/MacOS/RelWithDebInfo/libonnxruntime.pc ${PREFIX}/lib/pkgconfig/

cd ../..
rm -fr tmp
