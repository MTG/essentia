#!/usr/bin/env bash
#
# Build the static 3rdparty dependencies for the aarch64 Python wheels, inside a
# manylinux aarch64 container:
#
#   docker run --rm --volume "$PWD":/project --workdir /project \
#       quay.io/pypa/manylinux_2_28_aarch64:<tag> \
#       ./packaging/build_3rdparty_static_aarch64_manylinux.sh
#
# Mount the checkout at /project: the generated .pc files record absolute paths,
# and that is where cibuildwheel puts the project inside its build container.
#
# This is a thin container-setup wrapper around
# packaging/build_3rdparty_static_debian.sh; the actual per-library recipes are
# shared with every other platform.
set -euo pipefail

# The pinned sources are from 2010-2017 and manylinux_2_28 ships CMake 4, which
# removed things they still rely on: taglib 1.11.1 does
# `cmake_policy(SET CMP0022 OLD)`, and eigen 3.3.7 and chromaprint 1.5.1 declare
# a cmake_minimum_required below 3.5. Building them with the last CMake 3 keeps
# them byte-comparable with the x86_64 builder image, which also has a CMake 3,
# instead of patching four upstream tarballs.
CMAKE_VERSION=${CMAKE_VERSION:-3.31.10}
CMAKE_ROOT_DIR=/tmp/essentia-cmake

container_cmake_major=$(cmake --version 2>/dev/null | head -1 | cut -d' ' -f3 | cut -d. -f1)

if [ -z "${container_cmake_major}" ] || [ "${container_cmake_major}" -ge 4 ]; then
    echo "Container cmake is ${container_cmake_major:-absent}; installing cmake ${CMAKE_VERSION}"
    /opt/python/cp312-cp312/bin/python -m venv "${CMAKE_ROOT_DIR}/venv"
    "${CMAKE_ROOT_DIR}/venv/bin/pip" install --no-cache-dir --quiet "cmake==${CMAKE_VERSION}"

    # Expose only cmake itself, so the venv's python does not shadow the
    # container's python3.
    mkdir -p "${CMAKE_ROOT_DIR}/bin"
    ln -sf "${CMAKE_ROOT_DIR}/venv/bin/cmake" "${CMAKE_ROOT_DIR}/bin/cmake"
    export PATH="${CMAKE_ROOT_DIR}/bin:${PATH}"
fi

cmake --version | head -1

# The per-library recipes call plain `make`, which is serial. On a 4-vCPU
# arm64 runner that roughly triples the FFmpeg build.
export MAKEFLAGS="${MAKEFLAGS:--j$(nproc)}"

exec "$(dirname "$0")/build_3rdparty_static_debian.sh"
