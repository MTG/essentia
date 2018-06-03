#!/bin/bash

WRKDIR="$1"
PYBIN="$2"
PYBIN_PYTHON_MASK="$3"
PIPBIN="$4"
PIPBIN_PYTHON_MASK="$5"

if -z ${PYBIN};
then
   local PYBIN=/opt/python/cp36-cp36m/bin/
fi
if -z ${WRKDIR};
then
   local WRKDIR=/io
fi
if -z ${PYBIN_PYTHON_MASK};
then
   local PYBIN_PYTHON_MASK=/opt/python/*/bin
fi
if -z ${PIPBIN};
then
   local PIPBIN=${PYBIN}
fi
if -z ${PIPBIN_PYTHON_MASK};
then
   local PIPBIN_PYTHON_MASK=${PYBIN_PYTHON_MASK}
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
for PY_VER in python python3; do
   "${PYBIN}"/${PY_VER} waf configure --build-static --static-dependencies
   "${PYBIN}"/${PY_VER} waf
   "${PYBIN}"/${PY_VER} waf install
done

# cd - needs OLDPWD set
#cd -

# Compile wheels
for PIP_VER in PIP PIP3; do
#for PIP_VER in PIP; do
# use older version of numpy for backwards compatibility of its C API
    "${PIPBIN}"/${PIP_VER} install numpy==1.8.2
    ESSENTIA_WHEEL_SKIP_3RDPARTY=1 ESSENTIA_WHEEL_ONLY_PYTHON=1 "${PIPBIN}"/${PIP_VER} wheel ${WRKDIR}/ -w wheelhouse/ -v
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
for PIP_VER in PIP PIP3; do
    "${PIPBIN}"/${PIP_VER} install essentia --no-index -f ${WRKDIR}/wheelhouse
    for PY_VER in python python3; do
        (cd "$WRKDIR"; ${PYBIN}/${PY_VER} -c 'import essentia; import essentia.standard; import essentia.streaming')
    done
done
