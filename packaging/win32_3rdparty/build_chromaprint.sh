#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo "Building chromaprint $CHROMAPRINT_VERSION"

curl -SLO https://github.com/acoustid/chromaprint/releases/download/v$CHROMAPRINT_VERSION/chromaprint-$CHROMAPRINT_VERSION.tar.gz
tar -xf chromaprint-$CHROMAPRINT_VERSION.tar.gz
cd chromaprint-v$CHROMAPRINT_VERSION

CHROMAPRINT_TOOLCHAIN="
SET(CMAKE_SYSTEM_NAME Windows)
SET(CMAKE_C_COMPILER $HOST-gcc)
SET(CMAKE_CXX_COMPILER $HOST-g++)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
"
echo "$CHROMAPRINT_TOOLCHAIN" > $CHROMAPRINT_VERSION_toolchain.cmake


cmake \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$CHROMAPRINT_VERSION_toolchain.cmake \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TOOLS=OFF \
    -DFFMPEG_ROOT=$PREFIX \
    -DFFT_LIB=fftw3f \
    -DFFTW3_DIR=$PREFIX \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    .

make
make install

cd ../..
rm -r tmp

# patch libchromaprint.pc to add a missing link flag for fftw3f
sed -i -e 's/-lchromaprint/-lchromaprint -lfftw3f/g' "${PREFIX}"/lib/pkgconfig/libchromaprint.pc
