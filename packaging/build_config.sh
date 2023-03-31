#!/bin/sh

HOST=i686-w64-mingw32
if [ -z "${PREFIX}" ]; then
    PREFIX=`pwd`
fi
echo Installing to: $PREFIX

#SHARED_OR_STATIC="
#--enable-shared \
#--disable-static
#"

SHARED_OR_STATIC="
--disable-shared \
--enable-static
"

EIGEN_VERSION=3.3.7
FFMPEG_VERSION=ffmpeg-2.8.12
LAME_VERSION=3.100
TAGLIB_VERSION=taglib-1.11.1
ZLIB_VERSION=zlib-1.2.12
FFTW_VERSION=fftw-3.3.2
LIBSAMPLERATE_VERSION=libsamplerate-0.1.8
LIBYAML_VERSION=yaml-0.1.5
CHROMAPRINT_VERSION=1.4.3
QT_SOURCE_URL=https://download.qt.io/archive/qt/4.8/4.8.4/qt-everywhere-opensource-src-4.8.4.tar.gz
GAIA_VERSION=2.4.6-86-ged433ed
TENSORFLOW_VERSION=2.5.0


FFMPEG_AUDIO_FLAGS="
    --disable-programs
    --disable-doc
    --disable-debug

    --disable-avdevice
    --disable-avresample
    --disable-swscale
    --disable-postproc
    --disable-avfilter
    --enable-swresample

    --disable-network
    --disable-indevs
    --disable-outdevs
    --disable-muxers
    --disable-demuxers
    --disable-encoders
    --disable-decoders
    --disable-bsfs
    --disable-filters
    --disable-parsers
    --disable-protocols
    --disable-hwaccels

    --enable-protocol=file
    --enable-protocol=pipe

    --disable-sdl
    --disable-lzma
    --disable-zlib
    --disable-xlib
    --disable-bzlib
    --disable-libxcb

    --enable-demuxer=image2
    --enable-demuxer=aac
    --enable-demuxer=ac3
    --enable-demuxer=aiff
    --enable-demuxer=ape
    --enable-demuxer=asf
    --enable-demuxer=au
    --enable-demuxer=avi
    --enable-demuxer=flac
    --enable-demuxer=flv
    --enable-demuxer=matroska
    --enable-demuxer=mov
    --enable-demuxer=m4v
    --enable-demuxer=mp3
    --enable-demuxer=mpc
    --enable-demuxer=mpc8
    --enable-demuxer=ogg
    --enable-demuxer=pcm_alaw
    --enable-demuxer=pcm_mulaw
    --enable-demuxer=pcm_f64be
    --enable-demuxer=pcm_f64le
    --enable-demuxer=pcm_f32be
    --enable-demuxer=pcm_f32le
    --enable-demuxer=pcm_s32be
    --enable-demuxer=pcm_s32le
    --enable-demuxer=pcm_s24be
    --enable-demuxer=pcm_s24le
    --enable-demuxer=pcm_s16be
    --enable-demuxer=pcm_s16le
    --enable-demuxer=pcm_s8
    --enable-demuxer=pcm_u32be
    --enable-demuxer=pcm_u32le
    --enable-demuxer=pcm_u24be
    --enable-demuxer=pcm_u24le
    --enable-demuxer=pcm_u16be
    --enable-demuxer=pcm_u16le
    --enable-demuxer=pcm_u8
    --enable-demuxer=rm
    --enable-demuxer=shorten
    --enable-demuxer=tak
    --enable-demuxer=tta
    --enable-demuxer=wav
    --enable-demuxer=wv
    --enable-demuxer=xwma

    --enable-decoder=aac
    --enable-decoder=aac_latm
    --enable-decoder=ac3
    --enable-decoder=alac
    --enable-decoder=als
    --enable-decoder=ape
    --enable-decoder=atrac1
    --enable-decoder=atrac3
    --enable-decoder=eac3
    --enable-decoder=flac
    --enable-decoder=gsm
    --enable-decoder=gsm_ms
    --enable-decoder=mp1
    --enable-decoder=mp1float
    --enable-decoder=mp2
    --enable-decoder=mp2float
    --enable-decoder=mp3
    --enable-decoder=mp3float
    --enable-decoder=mp3adu
    --enable-decoder=mp3adufloat
    --enable-decoder=mp3on4
    --enable-decoder=mp3on4float
    --enable-decoder=mpc7
    --enable-decoder=mpc8
    --enable-decoder=ra_144
    --enable-decoder=ra_288
    --enable-decoder=ralf
    --enable-decoder=shorten
    --enable-decoder=tak
    --enable-decoder=truehd
    --enable-decoder=tta
    --enable-decoder=vorbis
    --enable-decoder=wavpack
    --enable-decoder=wmalossless
    --enable-decoder=wmapro
    --enable-decoder=wmav1
    --enable-decoder=wmav2
    --enable-decoder=wmavoice

    --enable-decoder=pcm_alaw
    --enable-decoder=pcm_bluray
    --enable-decoder=pcm_dvd
    --enable-decoder=pcm_f32be
    --enable-decoder=pcm_f32le
    --enable-decoder=pcm_f64be
    --enable-decoder=pcm_f64le
    --enable-decoder=pcm_lxf
    --enable-decoder=pcm_mulaw
    --enable-decoder=pcm_s8
    --enable-decoder=pcm_s8_planar
    --enable-decoder=pcm_s16be
    --enable-decoder=pcm_s16be_planar
    --enable-decoder=pcm_s16le
    --enable-decoder=pcm_s16le_planar
    --enable-decoder=pcm_s24be
    --enable-decoder=pcm_s24daud
    --enable-decoder=pcm_s24le
    --enable-decoder=pcm_s24le_planar
    --enable-decoder=pcm_s32be
    --enable-decoder=pcm_s32le
    --enable-decoder=pcm_s32le_planar
    --enable-decoder=pcm_u8
    --enable-decoder=pcm_u16be
    --enable-decoder=pcm_u16le
    --enable-decoder=pcm_u24be
    --enable-decoder=pcm_u24le
    --enable-decoder=pcm_u32be
    --enable-decoder=pcm_u32le
    --enable-decoder=pcm_zork

    --enable-parser=aac
    --enable-parser=aac_latm
    --enable-parser=ac3
    --enable-parser=cook
    --enable-parser=dca
    --enable-parser=flac
    --enable-parser=gsm
    --enable-parser=mlp
    --enable-parser=mpegaudio
    --enable-parser=tak
    --enable-parser=vorbis
    --enable-parser=vp3
    --enable-parser=vp8
"

FFMPEG_AUDIO_FLAGS_MUXERS="
    --enable-libmp3lame
    --enable-muxer=wav
    --enable-muxer=aiff
    --enable-muxer=mp3
    --enable-muxer=ogg
    --enable-muxer=flac
    --enable-encoder=pcm_s16le
    --enable-encoder=pcm_s16be
    --enable-encoder=libmp3lame
    --enable-encoder=vorbis
    --enable-encoder=flac
"

# see http://www.fftw.org/install/windows.html
FFTW_FLAGS="
    --enable-float \
    --enable-sse2 \
    --with-incoming-stack-boundary=2 \
    --with-our-malloc16
"

LIBSAMPLERATE_FLAGS="
    --disable-fftw \
    --disable-sndfile
"

QT_FLAGS="
    -no-accessibility
    -no-webkit
    -no-glib
    -no-xkb
    -no-xinput
    -no-fontconfig
    -no-mitshm
    -no-xrender
    -no-xrandr
    -no-xfixes
    -no-xcursor
    -no-xinerama
    -no-xsync
    -no-xvideo
    -no-xshape
    -no-sm
    -no-openvg
    -no-opengl
    -no-nas-sound
    -no-gtkstyle
    -no-dbus
    -no-pch
    -no-iconv
    -no-cups
    -no-nis
    -no-gui
    -no-openssl
    -no-libjpeg
    -no-libmng
    -no-libpng
    -no-libtiff
    -no-gif
    -no-scripttools
    -no-script
    -no-javascript-jit
    -no-svg
    -no-phonon-backend
    -no-phonon
    -no-audio-backend
    -no-multimedia
    -no-xmlpatterns
    -no-qt3support
    -qt-zlib
    -nomake demos
    -nomake examples
    -nomake tools
    -nomake translations
"

TENSORFLOW_FLAGS="
    --config=opt
    --config=monolithic
    --config=v2
    --config=noaws
    --config=nohdfs
    --config=nonccl
"

# The only known alternative to the interactive TensorFlow configuration is
# through env variables:
# https://github.com/tensorflow/tensorflow/issues/8527#issuecomment-289272898
#
# Set the required TensorFlow build env variables with CUDA support if they
# were not cofigured yet:
export PYTHON_BIN_PATH="${PYTHON_BIN_PATH:-python3}"
export USE_DEFAULT_PYTHON_LIB_PATH="${USE_DEFAULT_PYTHON_LIB_PATH:-1}"
export BAZEL_LINKLIBS="${BAZEL_LINKLIBS:--l%:libstdc++.a}"

export TF_NEED_JEMALLOC="${TF_NEED_JEMALLOC:-1}"
export TF_NEED_GCP="${TF_NEED_GCP:-0}"
export TF_NEED_HDFS="${TF_NEED_HDFS:-0}"
export TF_ENABLE_XLA="${TF_ENABLE_XLA:-0}"
export TF_NEED_OPENCL="${TF_NEED_OPENCL:-0}"

# TensorFlow CUDA versions intended for TensorFlow 2.5
# For future updates check the GPU compatibility chart:
# https://www.tensorflow.org/install/source#gpu
export TF_NEED_CUDA="${TF_NEED_CUDA:-1}"
export TF_CUDA_VERSION="${TF_CUDA_VERSION:-11.2}"
export TF_CUDNN_VERSION="${TF_CUDNN_VERSION:-8.1}"
export CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_PATH:-/usr/local/cuda}"
export CUDNN_INSTALL_PATH="${CUDNN_INSTALL_PATH:-/usr/local/cuda}"

# The compute capabilities define which GPUs can be used:
# https://developer.nvidia.com/cuda-gpus#compute
# Supporting more versions increases the library size, so
# for the moment it is set to a conservative number that
# covers some of the most popular dee'p learning GPUs:
# 3.5: Geforce GT XXX
# 5.2: Geforce GTX TITAN X
# 7.5: Geforce RTX 2080 (Ti)
# 8.6: Geforce RTX 30XX
export TF_CUDA_COMPUTE_CAPABILITIES="${TF_CUDA_COMPUTE_CAPABILITIES:-3.5,5.2,7.5,8.6}"
