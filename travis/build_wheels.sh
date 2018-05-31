#!/bin/bash

set -e -x

# Build tools
/io/travis/build_tools.sh

# Build static 3rdparty dependencies
/io/packaging/build_3rdparty_static_debian.sh

# Build static libessentia.a library
# Use Python3.6. CentOS 5's native python is too old...
PYBIN=/opt/python/cp36-cp36m/bin/
cd /io
"${PYBIN}/python" waf configure --build-static --static-dependencies
"${PYBIN}/python" waf
"${PYBIN}/python" waf install
cd -

# Compile wheels
for PYBIN in /opt/python/*/bin; do
# use older version of numpy for backwards compatibility of its C API
    "${PYBIN}/pip" install numpy==1.8.2
    ESSENTIA_WHEEL_SKIP_3RDPARTY=1 ESSENTIA_WHEEL_ONLY_PYTHON=1 "${PYBIN}/pip" wheel /io/ -w wheelhouse/ -v
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
# Do not run auditwheel for six package because of a bug
# https://github.com/pypa/python-manylinux-demo/issues/7
    if [[ "$whl" != wheelhouse/six* ]];
    then
        auditwheel repair "$whl" -w /io/wheelhouse/
    else
        cp "$whl" /io/wheelhouse/
    fi
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install essentia --no-index -f /io/wheelhouse
    (cd "$HOME"; ${PYBIN}/python -c 'import essentia; import essentia.standard; import essentia.streaming')
done