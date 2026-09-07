#!/usr/bin/env bash
#
# Install a prebuilt TensorFlow C library, its headers and a tensorflow.pc into a
# prefix, so that `./waf configure --with-tensorflow` finds it.
#
# The essentia-tensorflow wheels ship the TensorFlow C library inside the wheel;
# the user installs nothing else and needs no `tensorflow` pip package. Building
# TensorFlow from source on a CI runner is not viable, so the library comes from
# whichever upstream still publishes a prebuilt one for the platform in hand:
#
#   linux x86_64    Google tarball 2.18.1. manylinux2014 is glibc 2.17 and this
#                   tarball needs exactly that; the Homebrew bottle needs 2.27.
#   linux aarch64   Homebrew `libtensorflow` bottle 2.21.0, from ghcr.io. Google
#                   publishes no linux-arm64 libtensorflow, at any version.
#   macOS arm64     Homebrew `libtensorflow` bottle 2.21.0 (arm64_sequoia), from
#                   ghcr.io, built for macOS 15.0 since the formula started
#                   pinning its deployment target (Homebrew/homebrew-core#302294).
#   macOS x86_64    Google tarball 2.16.2, the last darwin-x86_64 release. Homebrew
#                   no longer builds an Intel macOS bottle for this formula.
#
# Every URL, checksum and version is pinned in packaging/build_config.sh and read
# from there, so this script has no version knowledge of its own. Downloads are
# verified against the pinned SHA256 before anything is unpacked.
#
# A macOS bottle is not usable as unpacked: it is bottled `cellar: :any`, so its
# dylib ids read `@@HOMEBREW_PREFIX@@/opt/libtensorflow/lib/<name>`, a placeholder
# `brew` rewrites when it pours. Linking against it as-is records that placeholder
# in the extension and delocate cannot resolve it. This script does what brew
# does: rewrites each id to its real location under <prefix>/lib, then re-signs
# ad hoc, because install_name_tool invalidates the signature and the arm64
# kernel kills any process that loads an invalidly signed library.
#
# The generated tensorflow.pc carries `-Wl,-rpath,${libdir}`, which is not
# decoration: without it the linker records a NEEDED on libtensorflow.so.2 /
# @rpath/libtensorflow.2.dylib that neither auditwheel nor delocate can resolve,
# and the repair step then silently ships a wheel with no TensorFlow in it (on
# Linux) or aborts (on macOS). The repair tools rewrite that rpath to a
# wheel-relative one, so the build-time path does not reach users.
#
# Usage:
#   packaging/fetch_libtensorflow.sh [--prefix DIR] [--platform PLATFORM]
#
#   --prefix    where to install (default /usr/local). <prefix>/lib/pkgconfig
#               must end up on PKG_CONFIG_PATH.
#   --platform  linux-x86_64 | linux-aarch64 | macos-arm64 | macos-x86_64
#               (default: detected from uname)

set -euo pipefail

PREFIX=/usr/local
PLATFORM=

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix|-p) PREFIX=$2; shift 2 ;;
        --platform)  PLATFORM=$2; shift 2 ;;
        -h|--help)   sed -n '3,32p' "$0"; exit 0 ;;
        *) echo "error: unknown argument $1" >&2; exit 2 ;;
    esac
done

script_dir=$(cd "$(dirname "$0")" && pwd)
config=$script_dir/build_config.sh

if [ ! -f "$config" ]; then
    echo "error: $config not found" >&2
    exit 1
fi

# Read one assignment out of build_config.sh. Sourcing it would be shorter but it
# is not side-effect free: it prints, exports a dozen TF_* build variables and
# runs nproc, which does not exist on macOS.
config_value() {
    sed -n "s/^$1=//p" "$config" | head -1
}

detect_platform() {
    case "$(uname -s)" in
        Darwin) os=macos ;;
        Linux)  os=linux ;;
        *) echo "error: unsupported operating system $(uname -s)" >&2; exit 1 ;;
    esac
    case "$(uname -m)" in
        arm64|aarch64) [ "$os" = macos ] && arch=arm64 || arch=aarch64 ;;
        x86_64|amd64)  arch=x86_64 ;;
        *) echo "error: unsupported machine $(uname -m)" >&2; exit 1 ;;
    esac
    echo "$os-$arch"
}

[ -n "$PLATFORM" ] || PLATFORM=$(detect_platform)

case "$PLATFORM" in
    linux-x86_64)  key=LINUX_X86_64  ;;
    linux-aarch64) key=LINUX_AARCH64 ;;
    macos-arm64)   key=MACOS_ARM64   ;;
    macos-x86_64)  key=MACOS_X86_64  ;;
    *) echo "error: unknown platform $PLATFORM" >&2; exit 1 ;;
esac

url=$(config_value "TENSORFLOW_${key}_URL")
sha256=$(config_value "TENSORFLOW_${key}_SHA256")
version=$(config_value "TENSORFLOW_${key}_VERSION")

if [ -z "$url" ] || [ -z "$sha256" ] || [ -z "$version" ]; then
    echo "error: $config has no complete pin for $PLATFORM" >&2
    exit 1
fi

echo "libtensorflow $version for $PLATFORM"
echo "  source: $url"

work=$(mktemp -d "${TMPDIR:-/tmp}/libtensorflow.XXXXXX")
trap 'rm -rf "$work"' EXIT

archive=$work/libtensorflow.tar.gz

# ghcr.io serves blobs only to a bearer token, but issues one to anybody who asks
# for pull scope on a public repository.
case "$url" in
    https://ghcr.io/v2/*)
        repository=${url#https://ghcr.io/v2/}
        repository=${repository%%/blobs/*}
        token=$(curl -fsSL "https://ghcr.io/token?service=ghcr.io&scope=repository:${repository}:pull" \
                | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')
        if [ -z "$token" ]; then
            echo "error: could not get an anonymous pull token for $repository" >&2
            exit 1
        fi
        curl -fsSL -H "Authorization: Bearer $token" -o "$archive" "$url"
        ;;
    *)
        curl -fsSL -o "$archive" "$url"
        ;;
esac

if command -v sha256sum > /dev/null 2>&1; then
    actual=$(sha256sum "$archive" | cut -d' ' -f1)
else
    actual=$(shasum -a 256 "$archive" | cut -d' ' -f1)
fi

if [ "$actual" != "$sha256" ]; then
    echo "error: checksum mismatch for $url" >&2
    echo "  expected $sha256" >&2
    echo "  actual   $actual" >&2
    exit 1
fi
echo "  sha256:  $actual (verified, $(wc -c < "$archive" | tr -d ' ') bytes)"

stage=$work/stage
mkdir -p "$stage"
tar xzf "$archive" -C "$stage"

# Google's tarball unpacks include/ and lib/ at the root; a Homebrew bottle nests
# them under libtensorflow/<version>/. Find the C API header and work back rather
# than encoding either layout.
header=$(find "$stage" -type f -path '*/include/tensorflow/c/c_api.h' | head -1)
if [ -z "$header" ]; then
    echo "error: no include/tensorflow/c/c_api.h in $url" >&2
    exit 1
fi
root=${header%/include/tensorflow/c/c_api.h}

# Any TensorFlow already in the prefix has to go first. The manylinux2014 x86_64
# builder image bakes in a Bazel-built libtensorflow 2.5 under /usr/local, and
# leaving it there lets the linker pick its libtensorflow_framework out from
# under the headers installed here.
rm -rf "$PREFIX"/lib/libtensorflow.so* "$PREFIX"/lib/libtensorflow_framework.so* \
       "$PREFIX"/lib/libtensorflow.*dylib "$PREFIX"/lib/libtensorflow_framework.*dylib \
       "$PREFIX"/lib/pkgconfig/tensorflow.pc \
       "$PREFIX"/include/tensorflow "$PREFIX"/include/tsl "$PREFIX"/include/xla

# A Homebrew bottle also ships lib/pkgconfig/tensorflow.pc, but with placeholder
# paths and without the rpath the comment at the top explains; the one written
# below replaces it, so it is not copied at all.
mkdir -p "$PREFIX/lib/pkgconfig" "$PREFIX/include"
cp -a "$root/include/." "$PREFIX/include/"
for entry in "$root"/lib/*; do
    case "$(basename "$entry")" in
        pkgconfig) continue ;;
    esac
    cp -a "$entry" "$PREFIX/lib/"
done

# A bottle's files unpack read-only (444) and cp -a keeps that. As root, which
# is what the Linux build containers run as, it makes no difference; as a user,
# which is what the macOS runners are, nothing could later overwrite them in
# place. Google's tarballs are already writable, so this is a no-op for them.
for entry in "$root"/include/* "$root"/lib/*; do
    [ -e "$PREFIX/${entry#"$root/"}" ] || continue
    chmod -R u+w "$PREFIX/${entry#"$root/"}"
done

# Relocate a Homebrew macOS bottle (see the header). Only regular files: the
# versioned dylib carries the id, the symlinks just point at it. Only ids whose
# value is a Homebrew placeholder, so Google's tarballs (id @rpath/<name>) and
# any already-relocated prefix are left alone. Neither library loads the other
# through a placeholder path (libtensorflow is monolithic here), so LC_LOAD_DYLIB
# entries need no rewriting; the loop asserts that rather than assuming it.
case "$PLATFORM" in
    macos-*)
        for lib in "$PREFIX"/lib/libtensorflow*.dylib; do
            [ -f "$lib" ] && [ ! -L "$lib" ] || continue
            id=$(otool -D "$lib" | sed -n '2p')
            case "$id" in
                @@HOMEBREW_*)
                    echo "  relocating $(basename "$lib"): $id"
                    install_name_tool -id "$PREFIX/lib/${id##*/}" "$lib"
                    codesign --force --sign - "$lib"
                    ;;
            esac
            if otool -L "$lib" | grep -q '@@HOMEBREW_'; then
                echo "error: $(basename "$lib") still references a Homebrew placeholder:" >&2
                otool -L "$lib" | grep '@@HOMEBREW_' >&2
                exit 1
            fi
        done
        ;;
esac

cat > "$PREFIX/lib/pkgconfig/tensorflow.pc" <<EOF
prefix=$PREFIX
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: tensorflow
Description: TensorFlow C API, prebuilt (packaging/fetch_libtensorflow.sh)
Version: $version
Requires:
Libs: -L\${libdir} -ltensorflow -ltensorflow_framework -Wl,-rpath,\${libdir}
Cflags: -I\${includedir}
EOF

echo "  installed to $PREFIX"
echo "  wrote $PREFIX/lib/pkgconfig/tensorflow.pc"

# Say enough about what landed that a CI log can be read without the artifacts.
library=$(find "$PREFIX/lib" -maxdepth 1 -type f \
    \( -name 'libtensorflow.so.*' -o -name 'libtensorflow.*.dylib' \) 2>/dev/null | head -1)
if [ -n "$library" ]; then
    echo "  library: $(basename "$library") ($(wc -c < "$library" | tr -d ' ') bytes)"
    # Whichever number decides whether this library is usable here: on macOS the
    # minimum OS version, which delocate propagates into the wheel tag, and on
    # Linux the glibc floor, which is what disqualifies a too-new build.
    case "$PLATFORM" in
        macos-*)
            command -v otool > /dev/null 2>&1 && \
                echo "  minimum macOS: $(otool -l "$library" | awk '/minos/ {print $2; exit}')"
            ;;
        linux-*)
            command -v readelf > /dev/null 2>&1 && \
                echo "  highest glibc required: $(readelf -V "$library" 2>/dev/null \
                    | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"
            ;;
    esac
fi

echo
echo "Add $PREFIX/lib/pkgconfig to PKG_CONFIG_PATH, then run:"
echo "    ./waf configure --with-tensorflow"
