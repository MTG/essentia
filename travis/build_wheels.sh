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

# To prevent errors parsing version with `git describe`.
git config --global --add safe.directory /io

# We are dropping support for Python 3.4 since PyYaml is not supporting it anymore.
# We can just remove the Python3.4 folder until ManyLinux1 drops the support too.
rm -rf /opt/python/cp34-cp34m
rm -rf /opt/python/cp35-cp35m

# Build static libessentia.a library
# Use Python3.6. CentOS 5's native python is too old...
PYBIN=/opt/python/cp36-cp36m/bin/

cd /io

if [[ $WITH_TENSORFLOW ]]; then
    PROJECT_NAME='essentia-tensorflow'
    "${PYBIN}/python" waf configure --with-gaia --with-tensorflow --build-static --static-dependencies --pkg-config-path="${PKG_CONFIG_PATH}"
else
    PROJECT_NAME='essentia'
    "${PYBIN}/python" waf configure --with-gaia --build-static --static-dependencies --pkg-config-path="${PKG_CONFIG_PATH}"
fi

"${PYBIN}/python" waf
"${PYBIN}/python" waf install
cd -

# Compile wheels
for PYBIN in /opt/python/cp3*/bin; do
    # Use the oldest version of numpy for each Python version
    # for backwards compatibility of its C API
    # https://github.com/numpy/numpy/issues/5888

    # Use `oldest-supported-numpy` as a reference.
    # https://github.com/scipy/oldest-supported-numpy/blob/main/setup.cfg

    # Previously used this as a reference:
    # https://github.com/MacPython/scikit-learn-wheels/blob/master/.travis.yml

    # Python 3.x
    if [[ $PYBIN == *"cp311"* ]]; then
        # FIXME Not supported by NumPy yet. Update ASAP.
        continue
    elif [[ $PYBIN == *"cp310"* ]]; then
        NUMPY_VERSION=1.21.4
    elif [[ $PYBIN == *"cp39"* ]]; then
        NUMPY_VERSION=1.19.3
    elif [[ $PYBIN == *"cp38"* ]]; then
        NUMPY_VERSION=1.17.4
    elif [[ $PYBIN == *"cp37"* ]]; then
        NUMPY_VERSION=1.14.5
    elif [[ $PYBIN == *"cp36"* ]]; then
        NUMPY_VERSION=1.11.3
    # elif [[ $PYBIN == *"cp34"* ]] || [[ $PYBIN == *"cp35"* ]]; then
    #     NUMPY_VERSION=1.9.3
    fi

    "${PYBIN}/pip" install numpy==$NUMPY_VERSION

    ESSENTIA_WHEEL_SKIP_3RDPARTY=1 ESSENTIA_WHEEL_ONLY_PYTHON=1 \
    ESSENTIA_PROJECT_NAME="${PROJECT_NAME}" "${PYBIN}/pip" wheel /io/ -w wheelhouse/

    # Bundle external shared libraries into the essentia wheel now because
    # the tensorflow libraries are specific for each package version
    for whl in wheelhouse/*.whl; do
        PYVERSION=$( echo "$PYBIN" |cut -d/ -f4 )

        if [[ "$whl" == wheelhouse/essentia*"${PYVERSION}"* ]];
        then
            auditwheel repair "$whl" -w /io/wheelhouse/
        fi
    done
done

# Bundle external shared libraries into the rest of wheels
for whl in wheelhouse/*.whl; do
    if [[ "$whl" != wheelhouse/essentia* ]]; then
    # The wheels that already have `manylinux*` or `none-any`
    # (not OS-specific and suitable to any processor architecture) tags do not need
    # to be repaired with auditwheel. This applies to this known bug for six package
    # https://github.com/pypa/python-manylinux-demo/issues/7
        if [[ "$whl" != wheelhouse/*manylinux* && "$whl" != wheelhouse/*-none-any* ]];
        then
            auditwheel repair "$whl" -w /io/wheelhouse/
        else
            cp "$whl" /io/wheelhouse/
        fi
    fi
done

# Install and test
for PYBIN in /opt/python/cp3*/bin/; do
    if [[ $PYBIN == *"cp311"* ]]; then
        # FIXME Not supported by NumPy yet. Skip.
        continue
    fi
    "${PYBIN}/pip" install "${PROJECT_NAME}" --no-index -f /io/wheelhouse
    if [[ $WITH_TENSORFLOW ]]; then
    # Test that essentia can be imported along with the latest TensorFlow for each Python version
        "${PYBIN}/pip" install tensorflow
        (cd "$HOME"; ${PYBIN}/python -c 'import essentia; import essentia.standard; import essentia.streaming; import tensorflow')
    else
        (cd "$HOME"; ${PYBIN}/python -c 'import essentia; import essentia.standard; import essentia.streaming')
    fi
done
