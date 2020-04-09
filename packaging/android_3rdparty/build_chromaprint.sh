#!/bin/sh
. ../build_config_android.sh

echo "Building chromaprint $CHROMAPRINT_VERSION"

rm -rf tmp
mkdir tmp
cd tmp

curl -SLO https://github.com/acoustid/chromaprint/releases/download/v$CHROMAPRINT_VERSION/chromaprint-$CHROMAPRINT_VERSION.tar.gz
tar -xf chromaprint-$CHROMAPRINT_VERSION.tar.gz
cd chromaprint-v$CHROMAPRINT_VERSION

alias cmake-android='cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_TOOLCHAIN_CMAKE'

cmake \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TOOLS=OFF \
    -DFFMPEG_ROOT=$PREFIX \
    -DFFT_LIB=kissfft \
    -DKISSFFT_SOURCE_DIR="vendor/kissfft" \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_TOOLCHAIN_CMAKE \
    -DANDROID_PLATFORM=$ANDROID_API \
    -DANDROID_ABI=$ANDROID_ABI_CMAKE \
    .

make
make install

cd ../..
rm -r tmp

