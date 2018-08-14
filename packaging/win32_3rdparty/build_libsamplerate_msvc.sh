#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

wget http://www.mega-nerd.com/SRC/$LIBSAMPLERATE_VERSION.tar.gz
tar -xf $LIBSAMPLERATE_VERSION.tar.gz
cd $LIBSAMPLERATE_VERSION

# Install required files from libsndfile 
LIBSNDFILE_VERSION=libsndfile-1.0.28-w64
wget http://essentia.upf.edu/documentation/downloads/packaging/win/$LIBSNDFILE_VERSION.tar.gz
tar -xf $LIBSNDFILE_VERSION.tar.gz
cp $LIBSNDFILE_VERSION/libsndfile-1.* .
cp $LIBSNDFILE_VERSION/sndfile.h Win32/

# Replace I386 to x64
sed -i 's/I386/x64/g' Win32/Makefile.msvc
sed -i 's/i386/x64/g' Win32/Makefile.msvc

#"C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
# Get rid of /usr/bin and /bin paths that were added by MSYS2 and force using MSVC's link.exe
export PATH=${PATH/\/usr\/bin:/}
export PATH=${PATH/\/bin:/}

./Make.bat

export PATH=/usr/bin:/bin:$PATH

cp src/samplerate.h $PREFIX/include
cp libsamplerate-0.dll libsamplerate-0.lib $PREFIX/lib

# Create a *.pc file
VERSION=`echo $LIBSAMPLERATE_VERSION | awk -F "-"  '{ print $2 }'`
echo "
prefix=../packaging/win32_3rdparty
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: samplerate
Description: An audio Sample Rate Conversion library
Requires: 
Version: $VERSION
Libs: -L\${libdir} -lsamplerate
Cflags: -I\${includedir} 
" > $PREFIX/lib/pkgconfig/samplerate.pc

cd ../..
rm -r tmp
