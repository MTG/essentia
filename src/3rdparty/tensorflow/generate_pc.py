import argparse
import os


if __name__ == "__main__":
    """Creates a pkg-config file for TensorFlow.
    It locates the TensorFlow Python package and
    tells compiler to link again de .so libs
    contained on it.
    """

    try:
        import tensorflow

    except:
        raise(Exception('Tensorflow is not installed'))


    tf_dir = os.path.dirname(tensorflow.__file__)
    tf_version = tensorflow.__version__
    include_dir = os.path.dirname(__file__)


    print('Found TesnorFlow in "{}"'.format(tf_dir))
    print('TensorFlow version: {}'.format(tf_version))


    text = ('Name: tensorflow\n'
    'Description: machine learning lib -- development files\n'
    'Version: {0}\n'
    'Libs: -L{1} -L{1}/python '
    '-l:_pywrap_tensorflow_internal.so -l:libtensorflow_framework.so.1\n'
    'Cflags: -I{1}/include/tensorflow/ '
    '-I{2}/src/3rdparty/tensorflow/\n').format(tf_version,
                                              tf_dir,
                                              include_dir)

    with open('tensorflow.pc', 'w') as f:
        f.write(text)
