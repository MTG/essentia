#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

wget http://taglib.github.io/releases/$TAGLIB_VERSION.tar.gz
tar -xf $TAGLIB_VERSION.tar.gz
cd $TAGLIB_VERSION/

TAGLIB_TOOLCHAIN="
SET(CMAKE_SYSTEM_NAME Windows)
SET(CMAKE_C_COMPILER $HOST-gcc)
SET(CMAKE_CXX_COMPILER $HOST-g++)
SET(CMAKE_RC_COMPILER $HOST-windres)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
"
echo "$TAGLIB_TOOLCHAIN" > $TAGLIB_VERSION_toolchain.cmake

cmake \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$TAGLIB_VERSION_toolchain.cmake \
    -DENABLE_STATIC=ON \
	. 
make
make install

cd ../..
rm -r tmp

TAGLIB_PC="
prefix=$PREFIX
exec_prefix=\${prefix}
includedir=\${prefix}/include
libdir=\${exec_prefix}/lib

Name: TagLib
Description: Audio meta-data library
Requires:
Version: 1.9.1
Cflags: -I\${includedir}
Libs: -L\${libdir} -ltag
"
echo "$TAGLIB_PC" > $PREFIX/lib/pkgconfig/taglib.pc

mv $PREFIX/bin/libtag.dll $PREFIX/lib
