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

# We are dropping support for Python 3.4 since PyYaml is not supporting it anymore.
# We can just remove the Python3.4 folder until ManyLinux1 drops the support too.
rm -rf /opt/python/cp34-cp34m

# Build static libessentia.a library
# Use Python3.6. CentOS 5's native python is too old...
PYBIN=/opt/python/cp36-cp36m/bin/

cd /io

if [[ $WITH_TENSORFLOW ]]; then
# Build essentia with tensorflow support using tensorflow 1.15.0 as it is the
# newest version supported by the C API. It is backwards compatible for 1.X.X
# https://www.tensorflow.org/guide/versions
# Tensroflow >= 2.0 do not support libtensorflow for now
# https://www.tensorflow.org/install/lang_c
    PROJECT_NAME='essentia-tensorflow'
    TENSORFLOW_VERSION=1.15.0

    "${PYBIN}/pip" install tensorflow==$TENSORFLOW_VERSION
    "${PYBIN}/python" src/3rdparty/tensorflow/setup_tensorflow.py -m python -c "${PREFIX}"
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
    # Don't build for python 3.8 while tensorflow doesn't create wheels for it
    # https://github.com/tensorflow/addons/issues/744
    if [[ $WITH_TENSORFLOW ]] && [[ $PYBIN == *"cp38"* ]]; then break; fi
    if [[ $WITH_TENSORFLOW ]] && [[ $PYBIN == *"cp39"* ]]; then break; fi

    if [[ $WITH_TENSORFLOW ]]; then
    # The minimum numpy version required by tensorflow is always greater than
    # the installed one. Install the oldest numpy supported by each tensorflow
    # to get the maximum fordwards compatibility
        NUMPY_VERSION=$( "${PYBIN}/pip" check tensorflow |grep tensorflow |grep numpy |grep ">=" |awk -F"[>=',]+" '//{print $2}' )

        if [[ ${NUMPY_VERSION} ]]; then
            echo "Got numpy ${NUMPY_VERSION} from the Tensorflow requirements"
            "${PYBIN}/pip" install numpy==$NUMPY_VERSION
        fi

        "${PYBIN}/pip" install tensorflow==$TENSORFLOW_VERSION

        # Make the tensorflow symbolic links point to the shared libraries
        # installed with the tensorflow wheel
        "${PYBIN}/python" /io/src/3rdparty/tensorflow/setup_tensorflow.py -m python -c "${PREFIX}"

    else
    # Use the oldest version of numpy for each Python version
    # for backwards compatibility of its C API
    # https://github.com/numpy/numpy/issues/5888
    # Build numpy versions used by scikit-learn:
    # https://github.com/MacPython/scikit-learn-wheels/blob/master/.travis.yml

        # Python 2.7
        NUMPY_VERSION=1.8.2

        # Python 3.x
        if [[ $PYBIN == *"cp39"* ]]; then
            NUMPY_VERSION=1.19.3
        elif [[ $PYBIN == *"cp38"* ]]; then
            NUMPY_VERSION=1.17.4
        elif [[ $PYBIN == *"cp37"* ]]; then
            NUMPY_VERSION=1.14.5
        elif [[ $PYBIN == *"cp36"* ]]; then
            NUMPY_VERSION=1.11.3
        elif [[ $PYBIN == *"cp34"* ]] || [[ $PYBIN == *"cp35"* ]]; then
            NUMPY_VERSION=1.9.3
        fi

        "${PYBIN}/pip" install numpy==$NUMPY_VERSION
    fi

    ESSENTIA_WHEEL_SKIP_3RDPARTY=1 ESSENTIA_WHEEL_ONLY_PYTHON=1 \
    ESSENTIA_PROJECT_NAME="${PROJECT_NAME}" ESSENTIA_TENSORFLOW_VERSION="${TENSORFLOW_VERSION}" \
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/

    # Bundle external shared libraries into the essentia wheel now because
    # the tensorflow libraries are especific for each package version
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
    # Skip essentia-tensorflow until it is available for Python 3.8
    if [[ $WITH_TENSORFLOW ]] && [[ $PYBIN == *"cp38"* ]]; then break; fi

    "${PYBIN}/pip" install "${PROJECT_NAME}" --no-index -f /io/wheelhouse
    if [[ $WITH_TENSORFLOW ]]; then
    # Test that essentia can be imported along with tensorflow
        (cd "$HOME"; ${PYBIN}/python -c 'import essentia; import essentia.standard; import essentia.streaming; import tensorflow')
    else
        (cd "$HOME"; ${PYBIN}/python -c 'import essentia; import essentia.standard; import essentia.streaming')
    fi
done
