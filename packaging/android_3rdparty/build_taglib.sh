#!/bin/sh
. ../build_config_android.sh

echo "Building taglib $TAGLIB_VERSION"

rm -rf tmp
mkdir tmp
cd tmp

curl -SLO http://taglib.github.io/releases/$TAGLIB_VERSION.tar.gz
tar -xf $TAGLIB_VERSION.tar.gz
cd $TAGLIB_VERSION/

alias cmake-android='cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_TOOLCHAIN_CMAKE'

cmake \
    -D CMAKE_CXX_FLAGS="-fPIC" \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DZLIB_ROOT=$PREFIX \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_TOOLCHAIN_CMAKE \
    -DANDROID_PLATFORM=$ANDROID_API \
    -DANDROID_ABI=$ANDROID_ABI_CMAKE \
	. 
make
# patch taglib.cp (missing -lz flag)
sed -i 's/-ltag/-ltag -lz/g' taglib.pc
make install

cd ../..
rm -r tmp

