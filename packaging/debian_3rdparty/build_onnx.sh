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
CONFIG=Release

if [ "$OS" = "Darwin" ]; then
    DIR_OS="MacOS"
    CMAKE_EXTRA_DEFINES+=' CMAKE_OSX_ARCHITECTURES=arm64'
    SUFFIX="${LIBONNXRUNTIME_VERSION}.dylib*"

    ./build.sh                            \
            --config $CONFIG              \
            --build_shared_lib            \
            --parallel                    \
            --skip_submodule_sync         \
            --skip_tests                  \
            --cmake_extra_defines $CMAKE_EXTRA_DEFINES
else
    DIR_OS="Linux"
    SUFFIX="so*"

    ./build.sh                            \
            --config $CONFIG              \
            --build_shared_lib            \
            --parallel                    \
            --compile_no_warning_as_error \
            --skip_submodule_sync         \
            --allow_running_as_root       \
            --skip_tests                  \
            --cmake_extra_defines $CMAKE_EXTRA_DEFINES
fi

# copying onnxruntime files
mkdir -p "${PREFIX}"/lib/pkgconfig/
mkdir -p "${PREFIX}"/include/onnxruntime/

cp build/$DIR_OS/$CONFIG/libonnxruntime.pc ${PREFIX}/lib/pkgconfig/
cp -r build/$DIR_OS/$CONFIG/libonnxruntime.$SUFFIX ${PREFIX}/lib/

cp include/onnxruntime/core/session/onnxruntime_cxx_inline.h ${PREFIX}/include/onnxruntime/
cp include/onnxruntime/core/session/onnxruntime_float16.h ${PREFIX}/include/onnxruntime/
cp include/onnxruntime/core/session/onnxruntime_c_api.h ${PREFIX}/include/onnxruntime/
cp include/onnxruntime/core/session/onnxruntime_cxx_api.h ${PREFIX}/include/onnxruntime/

cd ../..
rm -fr tmp
