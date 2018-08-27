#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo "Building chromaprint $CHROMAPRINT_VERSION"

curl -SLO https://github.com/acoustid/chromaprint/releases/download/v$CHROMAPRINT_VERSION/chromaprint-$CHROMAPRINT_VERSION.tar.gz
tar -xf chromaprint-$CHROMAPRINT_VERSION.tar.gz
cd chromaprint-v$CHROMAPRINT_VERSION

cmake \
    -G "Visual Studio 15 2017 Win64" \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TOOLS=OFF \
    -DFFT_LIB=fftw3f \
    -DFFTW3_DIR=$PREFIXX

msbuild.exe all_build.vcxproj -p:Configuration=Release
msbuild.exe install.vcxproj

mv $PREFIX/bin/chromaprint.dll $PREFIX/lib

# Create taglib.pc file from scratch 
VERSION=`echo $CHROMAPRINT_VERSION | awk -F "-"  '{ print $2 }'`
echo "
prefix=../packaging/win32_3rdparty
exec_prefix=\${prefix}

libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: chromaprint
Description: Audio meta-data library
Requires: 
Version: $VERSION
Libs: -L\${libdir} -lchromaprint -lfftw3f
Cflags: -I\${includedir}
" > $PREFIX/lib/pkgconfig/libchromaprint.pc

cd ../..
rm -r tmp
