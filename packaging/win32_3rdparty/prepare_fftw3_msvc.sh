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

mkdir include; mkdir lib
mv fftw3.h include/
mv libfftw3f-3.def lib/fftw3f.def
mv libfftw3f-3.exp lib/fftwf3.exp
mv libfftw3f-3.lib lib/fftw3f.lib
mv libfftw3f-3.dll lib/

cd ..
mkdir -p builds/$FFTW_VERSION
mv tmp/include builds/$FFTW_VERSION
mv tmp/lib builds/$FFTW_VERSION
rm -r tmp

# TODO: create pc file
