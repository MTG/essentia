#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

wget http://taglib.github.io/releases/$TAGLIB_VERSION.tar.gz
tar -xf $TAGLIB_VERSION.tar.gz
cd $TAGLIB_VERSION/

# For an x86 build, remove -G flag that is specifying the x64 generator
cmake -G "Visual Studio 15 2017 Win64" -DCMAKE_INSTALL_PREFIX=$PREFIX -DBUILD_SHARED_LIBS=ON -DZLIB_ROOT=$PREFIX
msbuild.exe all_build.vcxproj -p:Configuration=Release
msbuild.exe install.vcxproj

mv $PREFIX/bin/tag.dll $PREFIX/lib

# Create taglib.pc file from scratch 
VERSION=`echo $TAGLIB_VERSION | awk -F "-"  '{ print $2 }'`
echo "
prefix=../packaging/win32_3rdparty
exec_prefix=\${prefix}

libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: TagLib
Description: Audio meta-data library
Requires: 
Version: $VERSION
Libs: -L\${libdir} -ltag
Cflags: -I\${includedir}/taglib
" > $PREFIX/lib/pkgconfig/taglib.pc

cd ../..
rm -r tmp
