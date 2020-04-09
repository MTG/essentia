. ../build_config.sh

ANDROID_NDK=/home/dbogdanov/dev/android-ndk-r21
# Path to NDK

ANDROID_BUILD_ARCH=x86_64
# The CPU architecture of the NDK build tools (only x86_64 is available).
# Available platforms: https://developer.android.com/ndk/downloads

ANDROID_API=21
# Target Android API version. 
# See https://developer.android.com/ndk/guides/cmake#android_platform
# API 21: Android 5.0 Lollipop.

ANDROID_ARCH=arm
# Required by FFmpeg. Target architecture:
# - arm
# - aarch64
# - i686
# - x86_64

ANDROID_ABI_CMAKE=armeabi-v7a
# Required by CMake. Target Android ABI:
# - armeabi-v7a     
# - armeabi-v7a with NEON
# - arm64-v8a   
# - x86     
# - x86_64
# See https://developer.android.com/ndk/guides/cmake#android_abi

ANDROID_TARGET=armv7a-linux-androideabi
# Requiered by Autoconf. Target Android canonical triplet:
# - aarch64-linux-android
# - armv7a-linux-androideabi
# - i686-linux-android
# - x86_64-linux-android

ANDROID_TARGET_BINUTILS=$(echo $ANDROID_TARGET | sed "s/armv7a/arm/")
# Required for Autoconf. For 32-bit ARM (armv7a), binutils tools are prefixed with 'arm'.

ANDROID_TOOLCHAIN=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-$ANDROID_BUILD_ARCH
ANDROID_TOOLCHAIN_CMAKE=$ANDROID_NDK/build/cmake/android.toolchain.cmake
ANDROID_CROSS_PREFIX=$ANDROID_TOOLCHAIN/bin/$ANDROID_TARGET_BINUTILS

export CC=$ANDROID_TOOLCHAIN/bin/$ANDROID_TARGET$ANDROID_API-clang
export CXX=$ANDROID_TOOLCHAIN/bin/$ANDROID_TARGET$ANDROID_API-clang++

export AR=$ANDROID_CROSS_PREFIX-ar
export AS=$ANDROID_CROSS_PREFIX-as
export NM=$ANDROID_CROSS_PREFIX-nm
export LD=$ANDROID_CROSS_PREFIX-ld
export RANLIB=$ANDROID_CROSS_PREFIX-ranlib
export STRIP=$ANDROID_CROSS_PREFIX-strip

# This will overwrite the value from build_config.sh
SHARED_OR_STATIC="
--enable-shared \
--disable-static
"

# Requiers newer libyaml version for Android builds
LIBYAML_VERSION=yaml-0.2.2
