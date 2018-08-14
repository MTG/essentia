#!/bin/sh
BASEDIR=$(dirname $0)
cd $BASEDIR/win32_3rdparty
rm -rf bin dynamic include lib share

./prepare_fftw3_msvc.sh
./prepare_ffmpeg_msvc.sh
./build_libsamplerate_msvc.sh
./build_zlib_msvc.sh
./build_taglib_msvc.sh
./build_yaml.sh
