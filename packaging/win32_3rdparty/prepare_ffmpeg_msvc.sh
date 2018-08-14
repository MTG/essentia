#!/bin/sh

pacman -S tar make gcc diffutils --noconfirm
./build_ffmpeg_msvc.sh

cd lib/
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

# TODO patch .pc file here

cd ..

# TODO avoid writing to include/ lib/. Use prefix instead
# TODO use FFmpeg version variable
mkdir -p builds/ffmpeg-2.8.12
mv include builds/ffmpeg-2.8.12
mv lib builds/ffmpeg-2.8.12
