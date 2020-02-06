#!/bin/sh
. ../build_config.sh


rm -rf tmp
mkdir tmp
cd tmp


echo "Building Tensorflow $TENSORFLOW_VERSION"


curl -SLO https://github.com/tensorflow/tensorflow/archive/v$TENSORFLOW_VERSION.tar.gz
tar -xf v$TENSORFLOW_VERSION.tar.gz
cd tensorflow-$TENSORFLOW_VERSION


# Force using curl for the dependencies download
sed -i 's/\[\[ \"\$OSTYPE\" == \"darwin\"\* \]\]/true/g' tensorflow/contrib/makefile/download_dependencies.sh

tensorflow/contrib/makefile/download_dependencies.sh

# Add fPIC, otherwise nsync won't compile
sed -i 's/PLATFORM_CFLAGS=-std=c++11 -Werror -Wall -Wextra -pedantic/PLATFORM_CFLAGS=-std=c++11 -Wall -Wextra -pedantic -fPIC/g' tensorflow/contrib/makefile/compile_nsync.sh

# Compile the C API as it's curretly the recomended way to interface Tensorflow
sed -i 's/CORE_CC_ALL_SRCS := \\/&\n\$(wildcard tensorflow\/c\/c_api.cc) \\/' tensorflow/contrib/makefile/Makefile

# Define android to prevent errors in the Andrid API. Why?
sed -i 's/#ifndef __ANDROID__/#define __ANDROID__ 1\n&/' tensorflow/c/c_api.cc

# Use relative paths to prevent too long lines.
sed -i 's/"\$(cd "\$(dirname "\${BASH_SOURCE\[0]}")" \&\& pwd)"/"\$(dirname "\${BASH_SOURCE\[0]}")"/' tensorflow/contrib/makefile/build_all_linux.sh tensorflow/contrib/makefile/build_all_linux.sh

# We don't want to re-download dependencies on the build script as it would undo the previous patches
sed -i 's/rm -rf tensorflow\/contrib\/makefile\/downloads/# &/' tensorflow/contrib/makefile/build_all_linux.sh
sed -i 's/tensorflow\/contrib\/makefile\/download_dependencies.sh/# &/' tensorflow/contrib/makefile/build_all_linux.sh

# Add -lrt flag.
sed -i 's/HOST_CXXFLAGS=\"--std=c++11 -march=native\" \\/HOST_CXXFLAGS=\"--std=c++11 -march=native -lrt\" \\/' tensorflow/contrib/makefile/build_all_linux.sh

# Prevent compiling the example.
sed -i 's/all: \$(LIB_PATH) \$(BENCHMARK_NAME)/all: \$(LIB_PATH)/' tensorflow/contrib/makefile/Makefile

echo 'tensorflow/core/kernels/matmul_op_fused.cc' >> tensorflow/contrib/makefile/tf_op_files.txt

tensorflow/contrib/makefile/build_all_linux.sh

PREFIX_LIB=${PREFIX}/lib/tensorflow
PREFIX_INCLUDE=${PREFIX}/include/tensorflow/c

mkdir ${PREFIX_LIB}
mkdir -p ${PREFIX_INCLUDE}

cp tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a ${PREFIX_LIB}
cp tensorflow/contrib/makefile/gen/protobuf/lib/libprotobuf.a ${PREFIX_LIB}
cp tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11/libnsync.a ${PREFIX_LIB}

cp tensorflow/c/c_api.h ${PREFIX_INCLUDE}

./tensorflow/c/generate-pc.sh -p ${PREFIX} -l ${PREFIX_LIB} -v $TENSORFLOW_VERSION

echo "Generating pkgconfig file for TensorFlow $TENSORFLOW_VERSION in ${PREFIX}"

cat << EOF > ${PREFIX}/lib/pkgconfig/tensorflow.pc
prefix=${PREFIX}
exec_prefix=\${prefix}
libdir=\${prefix}/lib/tensorflow
includedir=\${prefix}/include/tensorflow/c
Name: TensorFlow
Version: ${TENSORFLOW_VERSION}
Description: Library for computation using data flow graphs for scalable machine learning
Requires:
Libs: -L\${libdir} -Wl,--allow-multiple-definition -Wl,--whole-archive -ltensorflow-core -lprotobuf -lnsync
Libs.private: -lz -lm -ldl -lpthread
Cflags: -I\${includedir}
EOF


cd ../..
# rm -r tmp
