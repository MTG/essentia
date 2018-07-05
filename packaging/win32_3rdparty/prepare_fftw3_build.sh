#!/bin/sh

pacman -S unzip --noconfirm
rm -rf tmp
mkdir tmp; cd tmp
wget ftp://ftp.fftw.org/pub/fftw/fftw-3.3.3-dll64.zip
unzip fftw-3.3.3-dll64.zip

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
mkdir -p builds/fftw-3.3.3
mv tmp/include builds/fftw-3.3.3
mv tmp/lib builds/fftw-3.3.3
rm -r tmp
