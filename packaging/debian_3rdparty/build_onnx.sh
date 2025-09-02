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

# Build the dynamic library for Linux or MacOS
# build for Intel and Apple silicon CPUs --> "x86_64;arm64"

CMAKE_EXTRA_DEFINES="FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER CMAKE_INSTALL_PREFIX=${PREFIX}"
OS=$(uname -s)

if [ "$OS" = "Darwin" ]; then
    DIR_OS="MacOS"
    CMAKE_EXTRA_DEFINES+=' CMAKE_OSX_ARCHITECTURES="arm64"'
else
    DIR_OS="Linux"
fi

CONFIG="Release"
./build.sh                                \
            --config $CONFIG              \
            --build_shared_lib            \
            --parallel                    \
            --compile_no_warning_as_error \
            --skip_submodule_sync         \
            --allow_running_as_root       \
            --skip_tests                  \
            --cmake_extra_defines $CMAKE_EXTRA_DEFINES

# copying .pc file
mkdir -p "${PREFIX}"/lib/pkgconfig/
cp -r build/$DIR_OS/$CONFIG/libonnxruntime.* ${PREFIX}/lib/pkgconfig/

cd ../..
rm -fr tmp
