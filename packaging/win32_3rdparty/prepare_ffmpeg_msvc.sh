#!/bin/sh
. ../build_config.sh

# There are FFmpeg builds available online (http://ffmpeg.zeranoe.com/builds/),
# but unfortunately those do not include libavresample. Therefore, we have to
# build FFmpeg from scratch.
./build_ffmpeg_msvc.sh

cd $PREFIX/lib
lib /def:avcodec-56.def /out:avcodec-56.lib
lib /def:avformat-56.def /out:avformat-56.lib
lib /def:avutil-54.def /out:avutil-54.lib
lib /def:avresample-2.def /out:avresample-2.lib
lib /def:swresample-1.def /out:swresample-1.lib

mv avcodec-56.lib avcodec.lib
mv avformat-56.lib avformat.lib
mv avutil-54.lib avutil.lib
mv avresample-2.lib avresample.lib
mv swresample-1.lib swresample.lib

sed -i 's/^prefix=.*/prefix=\.\.\/packaging\/win32_3rdparty/' pkgconfig/libav*.pc
