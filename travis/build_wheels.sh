#!/bin/bash
set -e -x

# Build tools if using original quay.io/pypa/manylinux1_* docker images
# (already built in mtgupf/essentia-builds images)
#/io/travis/build_tools.sh

# Build static 3rdparty dependencies
# (already built in mtgupf/essentia-builds images)
#/io/packaging/build_3rdparty_static_debian.sh --with-gaia

# Location of the dependencies in essentia-builds docker images
# ...is already set in the docker images
#PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
#PKG_CONFIG_PATH=/io/packaging/debian_3rdparty/lib/pkgconfig

# Build static libessentia.a library
# Use Python3.6. CentOS 5's native python is too old...
PYBIN=/opt/python/cp36-cp36m/bin/

cd /io
"${PYBIN}/python" waf configure --with-gaia --build-static --static-dependencies --pkg-config-path="${PKG_CONFIG_PATH}"
"${PYBIN}/python" waf
"${PYBIN}/python" waf install
cd -

# Compile wheels
for PYBIN in /opt/python/*/bin; do
# Use the oldest version of numpy for each Python version
# for backwards compatibility of its C API
# https://github.com/numpy/numpy/issues/5888
# Build numpy versions used by scikit-learn:
# https://github.com/MacPython/scikit-learn-wheels/blob/master/.travis.yml

# Python 2.7
    NUMPY_VERSION=1.8.2

# Python 3.x
    if [[ $PYBIN == *"cp37"* ]]; then
        NUMPY_VERSION=1.14.5
    elif [[ $PYBIN == *"cp36"* ]]; then
        NUMPY_VERSION=1.11.3
    elif [[ $PYBIN == *"cp34"* ]] || [[ $PYBIN == *"cp35"* ]]; then
        NUMPY_VERSION=1.9.3
    fi

    "${PYBIN}/pip" install numpy==$NUMPY_VERSION
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