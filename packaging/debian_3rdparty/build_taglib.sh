#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

wget http://taglib.github.io/releases/$TAGLIB_VERSION.tar.gz
tar -xf $TAGLIB_VERSION.tar.gz
cd $TAGLIB_VERSION/

cmake \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
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
#echo "$TAGLIB_PC" > $PREFIX/lib/pkgconfig/taglib.pc

