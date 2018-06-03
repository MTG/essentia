#!/bin/bash

if $1;
then
   # usually /usr/bin
   PYBIN = $1
   PYBIN_PYTHON = $1
else
   PYBIN=/opt/python/cp36-cp36m/bin/
   PYBIN_PYTHON = /opt/python/*/bin
fi
if $2
then
   WRKDIR = $2
else
   WRKDIR = /io
fi


set -e -x

# Build tools if using original quay.io/pypa/manylinux1_* docker images
# (already built in mtgupf/essentia-builds images)
#/io/travis/build_tools.sh

# Build static 3rdparty dependencies
${WRKDIR}/packaging/build_3rdparty_static_debian.sh

# Build static libessentia.a library
# Use Python3.6. CentOS 5's native python is too old...
#PYBIN=/opt/python/cp36-cp36m/bin/
cd ${WRKDIR}
"${PYBIN}/python" waf configure --build-static --static-dependencies
"${PYBIN}/python" waf
"${PYBIN}/python" waf install
cd -

# Compile wheels
for ${PYBIN} in ${PYBIN_PYTHON}; do
# use older version of numpy for backwards compatibility of its C API
    "${PYBIN}/pip" install numpy==1.8.2
    ESSENTIA_WHEEL_SKIP_3RDPARTY=1 ESSENTIA_WHEEL_ONLY_PYTHON=1 "${PYBIN}/pip" wheel ${WRKDIR}/ -w wheelhouse/ -v
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
# Do not run auditwheel for six package because of a bug
# https://github.com/pypa/python-manylinux-demo/issues/7
    if [[ "$whl" != wheelhouse/six* ]];
    then
        auditwheel repair "$whl" -w ${WRKDIR}/wheelhouse/
    else
        cp "$whl" ${WRKDIR}/wheelhouse/
    fi
done

# Install packages and test
for ${PYBIN} in ${PYBIN_PYTHON}; do
    "${PYBIN}/pip" install essentia --no-index -f ${WRKDIR}/wheelhouse
    (cd "$HOME"; ${PYBIN}/python -c 'import essentia; import essentia.standard; import essentia.streaming')
done
