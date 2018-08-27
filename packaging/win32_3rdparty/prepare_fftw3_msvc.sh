#!/bin/sh
. ../build_config.sh

pacman -S unzip --noconfirm
rm -rf tmp
mkdir tmp; cd tmp
wget ftp://ftp.fftw.org/pub/fftw/$FFTW_VERSION-dll64.zip
unzip $FFTW_VERSION-dll64.zip

lib /def:libfftw3-3.def
lib /def:libfftw3f-3.def
lib /def:libfftw3l-3.def

mkdir -p $PREFIX/include; mkdir -p $PREFIX/lib
mv fftw3.h $PREFIX/include/
mv libfftw3f-3.def $PREFIX/lib/fftw3f.def
mv libfftw3f-3.exp $PREFIX/lib/fftwf3.exp
mv libfftw3f-3.lib $PREFIX/lib/fftw3f.lib
mv libfftw3f-3.dll $PREFIX/lib/

# Create a *.pc file
VERSION=`echo $FFTW_VERSION | awk -F "-"  '{ print $2 }'`
echo "
prefix=../packaging/win32_3rdparty
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: FFTW
Description: fast Fourier transform library
Version: $VERSION
Libs: -L\${libdir} -lfftw3f
Cflags: -I\${includedir}
" > $PREFIX/lib/pkgconfig/fftw3f.pc

cd ..
rm -r tmp