#!/usr/bin/env bash
set -e
. ../build_config.sh

# rm -rf tmp
# mkdir tmp
cd tmp

# Prerequisites:        python>=3.10
#                       cmake>=3.28

echo "Building onnxruntime $LIBONNXRUNTIME_VERSION"

#curl -SLO "https://github.com/microsoft/onnxruntime/archive/refs/tags/v$LIBONNXRUNTIME_VERSION.tar.gz"
#! this file has an issue https://github.com/microsoft/onnxruntime/issues/24861
#! it is fixed manually for testing https://github.com/microsoft/onnxruntime/commit/f57db79743c4d1a3553aa05cf95bcd10966030e6
#! should be done in three-steps: first, downloading the package, editing the deps.txt file and running the script
#! in the main branch it is fixed, it should be replaced when a release is available.

#tar -xf v$LIBONNXRUNTIME_VERSION.tar.gz
cd onnxruntime-$LIBONNXRUNTIME_VERSION

python3 -m pip install cmake

# compile library with cmake
./build.sh \
  --config RelWithDebInfo \
  --build_shared_lib \
  --parallel \
  --compile_no_warning_as_error \
  --skip_submodule_sync

cd ../..
# rm -fr tmp
