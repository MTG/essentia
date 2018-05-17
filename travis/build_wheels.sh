#!/bin/bash

set -e -x

# yasm on CentOS 5 is too old, install a newer version
wget http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
tar -xvf yasm-1.3.0.tar.gz
cd yasm-1.3.0
./configure
make
make install
cd ..

# cmake is also too old, taglib requires CMake 2.8.0
# use curl; there's a SSL certificate error with wget
curl -L --remote-name http://www.cmake.org/files/v2.8/cmake-2.8.10.2.tar.gz
tar -xvf cmake-2.8.10.2.tar.gz
cd cmake-2.8.10.2
./configure --prefix=/usr/local/cmake-2.8.10.2
make
make install
PATH=/usr/local/cmake-2.8.10.2/bin:$PATH
cd ..

function lex_pyver {
     # Echoes Python version string padded with zeros
     # Thus:
     # 3.2.1 -> 003002001
     # 3     -> 003000000
     echo $1 | awk -F "." '{printf "%03d%03d%03d", $1, $2, $3}'
}


for PYBIN in /opt/python/*/bin; do

# Patch python-config scripts (https://github.com/pypa/manylinux/pull/87)
# Remove -lpython from the python-config script.

    if [ -e ${PYBIN}/python3 ]; then
        ln -sf python3 ${PYBIN}/python
        ln -sf python3-config ${PYBIN}/python-config
    fi

    py_ver="$(${PYBIN}/python -c 'import platform; print(platform.python_version())')"

    if [ $(lex_pyver $py_ver) -lt $(lex_pyver 3.4) ]; then
        echo "Patching python 2"
        sed -i "s/'-lpython' *+ *pyver\( *+ *sys.abiflags\)\?/''/" $(readlink -e ${PYBIN}/python-config)
    else
        echo "Patching python 3"
        sed -i 's/-lpython${VERSION}${ABIFLAGS}//' $(readlink -e ${PYBIN}/python-config)
    fi
done


# Build static 3rdparty dependencies
/io/packaging/build_3rdparty_static_debian.sh

# Compile wheels
for PYBIN in /opt/python/*/bin; do
# use older version of numpy for backwards compatibility of its C API
    "${PYBIN}/pip" install numpy==1.8.2
    ESSENTIA_WHEEL_SKIP_3RDPARTY=1 "${PYBIN}/pip" wheel /io/ -w wheelhouse/ -v
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
# Do not run auditwheel for six package because of a bug
# https://github.com/pypa/python-manylinux-demo/issues/7
    if [[ "$whl" != wheelhouse/six* ]];
    then
        auditwheel repair "$whl" -w /io/wheelhouse/
    fi
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install essentia --no-index -f /io/wheelhouse
    (cd "$HOME"; ${PYBIN}/python -c 'import essentia; import essentia.standard; import essentia.streaming')
done