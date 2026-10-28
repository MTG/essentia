#!/usr/bin/env bash
set -e
. ../build_config.sh

echo "Building $FFMPEG_VERSION"

mux=$1
if test "$1" = "--no-muxers"; then
  echo Building FFmpeg without muxers
  FFMPEG_AUDIO_FLAGS_MUXERS=""
fi

rm -rf tmp
mkdir tmp
cd tmp

curl -SLO https://ffmpeg.org/releases/$FFMPEG_VERSION.tar.gz
tar xf $FFMPEG_VERSION.tar.gz
cd $FFMPEG_VERSION

./configure \
  --enable-pic \
  $FFMPEG_AUDIO_FLAGS \
  $FFMPEG_AUDIO_FLAGS_MUXERS \
  --prefix=$PREFIX \
  --extra-ldflags="-L$PREFIX/lib" \
  --extra-cflags="-I$PREFIX/include" \
  $SHARED_OR_STATIC
make
make install

# On AArch64 FFmpeg links against libatomic for the 128-bit atomics it uses and
# records -latomic in libavutil.pc, which propagates into essentia.pc and into
# the Python extension. libatomic.so.1 is on the manylinux policy whitelist, so
# auditwheel leaves it as an external dependency -- but the Debian slim images
# that most arm64 deployments are built on do not ship libatomic1, and importing
# the wheel there dies with "libatomic.so.1: cannot open shared object file".
# Ask for the static libatomic instead; nothing else in the wheel is dynamic.
case "$(uname -m)" in
  aarch64|arm64)
    sed -i -e 's/-latomic/-l:libatomic.a/g' "$PREFIX"/lib/pkgconfig/lib*.pc
    ;;
esac

cd ../..
rm -r tmp
