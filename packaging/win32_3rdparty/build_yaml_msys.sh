#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo $PREFIX

wget http://pyyaml.org/download/libyaml/$LIBYAML_VERSION.tar.gz
tar -xf $LIBYAML_VERSION.tar.gz
cd $LIBYAML_VERSION

# Use mingw64 instead of msys2's gcc for a native dll 
PATH=/mingw64/bin:$PATH

# Force to generate the def file with CFLAGS
CFLAGS=-Wl,--output-def=yaml.def ./configure \
    --prefix=$PREFIX \
    --host=x86_64-w64-mingw32 
#    --enable-shared --disable-static \
make
make install

mv $PREFIX/bin/libyaml-0-2.dll $PREFIX/lib/yaml.dll
rm $PREFIX/lib/libyaml.*

#dlltool $PREFIX/lib/yaml.dll -z yaml.def --export-all-symbols
lib -machine:X64 -def:src/yaml.def  -out:$PREFIX/lib/yaml.lib
sed -i 's/^prefix=.*/prefix=\.\.\/packaging\/win32_3rdparty/' $PREFIX/lib/pkgconfig/yaml-0.1.pc

#cd ../..
#rm -r tmp
