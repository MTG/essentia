#!/bin/sh

HOST=i686-w64-mingw32
if [ -z "${PREFIX}" ]; then
  PREFIX=$(pwd)
fi
echo Installing to: $PREFIX

SHARED_OR_STATIC="
--disable-shared
--enable-static
"

EIGEN_VERSION=3.3.7
FFMPEG_VERSION=ffmpeg-7.1.1
LAME_VERSION=3.100
TAGLIB_VERSION=taglib-1.11.1
ZLIB_VERSION=zlib-1.2.12
# FFTW 3.3.2 (2012) predates AArch64: its bundled config.sub does not know the
# aarch64-*-linux-gnu triplet and it has no NEON codelets for 64-bit ARM. Use
# 3.3.10 there. x86_64 and win32 keep 3.3.2 so their wheels are unchanged.
case "$(uname -m)" in
  aarch64|arm64) FFTW_VERSION=fftw-3.3.10 ;;
  *)             FFTW_VERSION=fftw-3.3.2 ;;
esac
LIBSAMPLERATE_VERSION=libsamplerate-0.1.9
LIBYAML_VERSION=yaml-0.1.5
CHROMAPRINT_VERSION=1.5.1
QT_SOURCE_URL=https://download.qt.io/archive/qt/4.8/4.8.4/qt-everywhere-opensource-src-4.8.4.tar.gz
GAIA_VERSION=2.4.6-86-ged433ed
# The version the Bazel build below compiles. Read by
# packaging/debian_3rdparty/build_tensorflow.sh and by the CUDA block further
# down, which was written against it. Nothing else reads it.
TENSORFLOW_VERSION=2.17.0

# The four pins below are what packaging/fetch_libtensorflow.sh installs when a
# wheel is built. They are independent of TENSORFLOW_VERSION and of each other:
# no single TensorFlow release publishes a prebuilt C library for all four
# platforms, so each names the newest release that works there and says why.
# Every pin the wheels use lives here and nowhere else.

# Linux x86_64: Google's official tarball rather than the Homebrew bottle. The
# bottle is newer but is built against glibc 2.27, and the manylinux2014 image
# these wheels come from is glibc 2.17, so the configure-time C API link test
# rejects it. The tarball's highest versioned symbol requirement is GLIBC_2.17,
# exactly what that image provides.
TENSORFLOW_LINUX_X86_64_VERSION=2.18.1
TENSORFLOW_LINUX_X86_64_URL=https://storage.googleapis.com/tensorflow/versions/2.18.1/libtensorflow-cpu-linux-x86_64.tar.gz
TENSORFLOW_LINUX_X86_64_SHA256=b692795f3ad198c531b02aeb2bc8146568d24aaf6a5dbf5faa43907c4028fd73

# Linux aarch64: a Homebrew bottle, pulled anonymously from the GitHub Container
# Registry, because Google has never published a Linux arm64 libtensorflow under
# any name. It needs at most GLIBC_2.27, which the manylinux_2_28 image used for
# this architecture provides. A bottle blob is addressed by its own digest, so
# the URL and the checksum carry the same value; the fetch script verifies it
# after downloading regardless. This is the `sha256` entry for arm64_linux in the
# libtensorflow formula's bottle block (2.21.0, rebuild 1).
TENSORFLOW_LINUX_AARCH64_VERSION=2.21.0
TENSORFLOW_LINUX_AARCH64_URL=https://ghcr.io/v2/homebrew/core/libtensorflow/blobs/sha256:b761769c0db1fa6602f8598e75e8253712af2920d77233e86e713e9293484444
TENSORFLOW_LINUX_AARCH64_SHA256=b761769c0db1fa6602f8598e75e8253712af2920d77233e86e713e9293484444

# macOS arm64: the Homebrew bottle as well, the `sha256` entry for arm64_sequoia
# in the same bottle block (2.21.0, rebuild 1). What decides this pin is the
# minimum macOS the library was built for, because delocate propagates the
# highest one it vendors into the wheel tag. Until Homebrew/homebrew-core#302294
# the formula let Bazel default that to the builder's SDK, so the Sequoia bottle
# carried 26.2 and the old `brew install tensorflow` step dragged the wheel up to
# macosx_26_2; the formula now pins it to the bottle's own tag, and this bottle
# reports minos 15.0 on every Mach-O in the keg, exactly the 15.0 the wheels
# target. Google's darwin-arm64 tarball (2.18.1, minos 12.0) would do too, but
# that channel stopped at 2.18 and the bottle is what linux/aarch64 already uses.
# The bottle's install names carry Homebrew's @@HOMEBREW_PREFIX@@ placeholder;
# the fetch script rewrites them, see there.
TENSORFLOW_MACOS_ARM64_VERSION=2.21.0
TENSORFLOW_MACOS_ARM64_URL=https://ghcr.io/v2/homebrew/core/libtensorflow/blobs/sha256:4646a8c41b89819998e8dd6295059a05ac10aa0b673c49f5538095170690f176
TENSORFLOW_MACOS_ARM64_SHA256=4646a8c41b89819998e8dd6295059a05ac10aa0b673c49f5538095170690f176

# macOS x86_64: Google's official tarball, built at 10.15, because there is no
# Homebrew bottle to pin. The rebuild that fixed the deployment target dropped
# the Intel `sonoma` bottle and Homebrew builds no x86_64 macOS bottles for this
# formula any more, so `brew install libtensorflow` on an Intel Mac would mean a
# multi-hour Bazel build. 2.16.2 is the last darwin-x86_64 tarball published.
TENSORFLOW_MACOS_X86_64_VERSION=2.16.2
TENSORFLOW_MACOS_X86_64_URL=https://storage.googleapis.com/tensorflow/versions/2.16.2/libtensorflow-cpu-darwin-x86_64.tar.gz
TENSORFLOW_MACOS_X86_64_SHA256=26b17967afbe99ef89c16f59b366d62b14c55c5c583af6e70aed8c3b3147ee9f

FFMPEG_AUDIO_FLAGS="
    --disable-programs
    --disable-doc
    --disable-debug

    --disable-avdevice
    --disable-swresample
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

    --disable-sdl2
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
# --enable-sse2, --with-incoming-stack-boundary and --with-our-malloc16 are
# x86-only and are rejected by configure on AArch64, where NEON is mandatory
# and malloc is already 16-byte aligned.
case "$(uname -m)" in
  aarch64|arm64)
    FFTW_FLAGS="
    --enable-float
    --enable-neon
"
    ;;
  *)
    FFTW_FLAGS="
    --enable-float
    --enable-sse2
    --with-incoming-stack-boundary=2
    --with-our-malloc16
"
    ;;
esac

# Several of the pinned source tarballs ship an autotools config.guess/config.sub
# that predates AArch64 -- libsamplerate 0.1.9 carries a 2009 copy, libyaml 0.1.5
# a 2010 one -- so their configure aborts with "unable to guess system type" on
# arm64. Refresh those two files in the unpacked tree from the build machine's
# own automake. Deliberately a no-op off arm64, so x86_64 and win32 builds keep
# using exactly the scripts their tarballs ship.
refresh_autotools_config() {
    case "$(uname -m)" in
        aarch64|arm64) ;;
        *) return 0 ;;
    esac

    for name in config.guess config.sub; do
        newest=$(ls /usr/local/share/automake-*/$name /usr/share/automake-*/$name 2>/dev/null | tail -1)
        if [ -z "$newest" ]; then
            echo "warning: no replacement $name found; leaving ${1:-.} alone" >&2
            continue
        fi
        for target in $(find "${1:-.}" -name "$name" -type f); do
            echo "Refreshing $target from $newest"
            cp "$newest" "$target"
            chmod +x "$target"
        done
    done
}

LIBSAMPLERATE_FLAGS="
    --disable-fftw
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
    --local_ram_resources=HOST_RAM*.6
    --jobs=$(nproc)
"

# The only known alternative to the interactive TensorFlow configuration is
# through env variables:
# https://github.com/tensorflow/tensorflow/issues/8527#issuecomment-289272898
#
# Set the required TensorFlow build env variables with CUDA support if they
# were not cofigured yet:
export PYTHON_BIN_PATH="${PYTHON_BIN_PATH:-$(which python3)}"
export USE_DEFAULT_PYTHON_LIB_PATH="${USE_DEFAULT_PYTHON_LIB_PATH:-1}"
export BAZEL_LINKLIBS="${BAZEL_LINKLIBS:--l%:libstdc++.a}"

export TF_NEED_JEMALLOC="${TF_NEED_JEMALLOC:-1}"
export TF_NEED_GCP="${TF_NEED_GCP:-0}"
export TF_NEED_HDFS="${TF_NEED_HDFS:-0}"
export TF_ENABLE_XLA="${TF_ENABLE_XLA:-0}"
export TF_NEED_OPENCL="${TF_NEED_OPENCL:-0}"
export TF_NEED_ROCM=0

# TensorFlow CUDA versions intended for TensorFlow 2.17.0
# For future updates check the GPU compatibility chart:
# https://www.tensorflow.org/install/source#gpu
export TF_NEED_CUDA="${TF_NEED_CUDA:-1}"
export TF_CUDA_VERSION="${TF_CUDA_VERSION:-12}"
export TF_CUDNN_VERSION="${TF_CUDNN_VERSION:-9}"
export CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_PATH:-/usr/local/cuda}"
export CUDNN_INSTALL_PATH="${CUDNN_INSTALL_PATH:-/usr/local/cuda}"

# The compute capabilities define which GPUs can be used:
# https://developer.nvidia.com/cuda-gpus#compute
# Supporting more versions increases the library size, so
# for the moment it is set to a conservative number that
# covers some of the most popular dee'p learning GPUs:
# 7.5: Geforce RTX 2080 (Ti)
# 8.0: Geforce RTX 3090 - 3080 (Ti)
# 8.6: Geforce RTX 30XX
# 8.9: Geforce RTX 4080
export TF_CUDA_COMPUTE_CAPABILITIES="${TF_CUDA_COMPUTE_CAPABILITIES:-7.5,8.0,8.6,8.9}"

# Silence interactive configure questions
export TF_SET_ANDROID_WORKSPACE=0
