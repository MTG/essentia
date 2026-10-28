#!/bin/bash
# Install a pinned sccache (https://github.com/mozilla/sccache) release binary.
#
# Used by the CI builds that run inside the manylinux2014 containers, where no
# package manager provides sccache. The statically linked musl builds run on
# any glibc, including the 2.17 shipped by those images.
#
# On macOS use `brew install sccache` instead (the cibuildwheel configs do).
set -euo pipefail

SCCACHE_VERSION=0.17.0
SCCACHE_PREFIX="${SCCACHE_PREFIX:-/usr/local/bin}"

if [[ "$(uname -s)" != Linux ]]; then
    echo "install_sccache.sh: only Linux is supported (use 'brew install sccache' on macOS)" >&2
    exit 1
fi

if command -v sccache >/dev/null 2>&1 && sccache --version | grep -q " ${SCCACHE_VERSION}\$"; then
    echo "sccache ${SCCACHE_VERSION} is already installed at $(command -v sccache)"
    exit 0
fi

# Checksums of the release tarballs, taken from the .sha256 assets published
# alongside them. Bump these together with SCCACHE_VERSION.
case "$(uname -m)" in
    x86_64)
        TARGET=x86_64-unknown-linux-musl
        SHA256=67c4a96dd237c1f518f6b36083f270f9976d516f1e57fce891755ea782e50006 ;;
    i686|i386)
        TARGET=i686-unknown-linux-musl
        SHA256=07cb06858d70e6d91678b1e8ed347c880ffd285c90ee4de3018384f1093c0dbf ;;
    aarch64|arm64)
        TARGET=aarch64-unknown-linux-musl
        SHA256=821a86343191aa1cbab74bd42f9e93c9a63bf85e4742945f40d3ae84193c1c77 ;;
    *)
        echo "install_sccache.sh: unsupported architecture $(uname -m)" >&2
        exit 1 ;;
esac

NAME="sccache-v${SCCACHE_VERSION}-${TARGET}"
URL="https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/${NAME}.tar.gz"

TMPDIR_SCCACHE="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_SCCACHE}"' EXIT

echo "Downloading ${URL}"
curl -fsSL --retry 3 -o "${TMPDIR_SCCACHE}/${NAME}.tar.gz" "${URL}"
echo "${SHA256}  ${TMPDIR_SCCACHE}/${NAME}.tar.gz" | sha256sum -c -

tar -xzf "${TMPDIR_SCCACHE}/${NAME}.tar.gz" -C "${TMPDIR_SCCACHE}"
install -m 0755 "${TMPDIR_SCCACHE}/${NAME}/sccache" "${SCCACHE_PREFIX}/sccache"

"${SCCACHE_PREFIX}/sccache" --version
