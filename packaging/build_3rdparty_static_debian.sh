#!/usr/bin/env bash
set -e
BASEDIR=$(dirname $0)
cd $BASEDIR/debian_3rdparty
./build_eigen3.sh
./build_fftw3.sh
./build_lame.sh
./build_ffmpeg.sh
./build_libsamplerate.sh
./build_zlib.sh
./build_taglib.sh
./build_yaml.sh
./build_chromaprint.sh

#!/usr/bin/env bash
if [[ "$*" == *--with-gaia* ]]
    then
        ./build_qt.sh
        ./build_gaia.sh
        rm -rf mkspecs plugins translations
fi

if [[ "$*" == *--with-tensorflow* ]]
    then
        ./build_tensorflow.sh
fi

rm -rf bin share
