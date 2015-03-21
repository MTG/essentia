#!/bin/sh
BASEDIR=$(dirname $0)
cd $BASEDIR/win32_3rdparty
rm -rf bin dynamic include lib share

./build_fftw3.sh  
./build_libav_nomuxers.sh  
./build_libsamplerate.sh  
./build_taglib.sh  
./build_yaml.sh
