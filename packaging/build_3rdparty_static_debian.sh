#!/bin/sh
BASEDIR=$(dirname $0)
cd $BASEDIR/debian_3rdparty
./build_fftw3.sh  
./build_libav_nomuxers.sh  
./build_libsamplerate.sh  
./build_taglib.sh  
./build_yaml.sh

if [ "$1" = --with-gaia ];
    then
        ./build_qt.sh
        ./build_gaia.sh
fi
