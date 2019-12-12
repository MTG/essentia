import argparse
from os import symlink, remove
from os.path import join, dirname, abspath
from shutil import copytree, rmtree
from subprocess import call


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Sets up tensorflow for linking against it.'
        'This can be done in two ways:'
        ' - python: By symlinking to an existing tensorflow python package. '
        '     This is the recommended way if tensorflow is already being used from python.'
        ' - libtensorflow: By downloading and installng the c API. This mode does not '
        '     allow simultaneous use of essentia and tensorflow from python.')

    parser.add_argument('--mode', '-m', default='python',
                        choices=['python', 'libtensorflow'])
    parser.add_argument('--platform', '-p', default='linux',
                        choices=['linux', 'macos', 'windows'])
    parser.add_argument('--context', '-c', default='/usr/local/')
    parser.add_argument('--version', '-v', default='1.14.0')
    parser.add_argument('--with_gpu', action='store_true')

    args = parser.parse_args()

    context = args.context

    if args.mode == 'python':
        print('looking for the tensorflow python package...')
        try:
            import tensorflow
        except ImportError as error:
            print(error.__class__.__name__ + ": " + error.message)
            raise(Exception('tensorflow is not available from this interpreter.\n'
                            'Suggestion: `pip install tensorflow`'))

        tf_dir = dirname(tensorflow.__file__)
        version = tensorflow.__version__

        version_list = version.split('.')
        major_version = int(version_list[0])
        minor_version = int(version_list[1])

        # From Tensorflow 1.15. libraries are stored in `tensorflow_core`
        if major_version >= 1:
            if minor_version >= 15:
                tf_dir = tf_dir + '_core'

        print('found tensorflow in "{}"'.format(tf_dir))
        print('tensorflow version: {}'.format(version))

        file_dir = dirname(abspath(__file__))

        # create symbolic links fo the libraries
        print('creating symbolic links...')

        src = join(tf_dir, 'libtensorflow_framework.so.1')
        tgt = join(context, 'lib', 'libtensorflow_framework.so')
        call(['ln', '-sf', src, tgt])
        call(['ln', '-sf', 'libtensorflow_framework.so', join(context, 'lib', 'libtensorflow_framework.so.1')])

        src = join(tf_dir, 'python', '_pywrap_tensorflow_internal.so')
        tgt = join(context, 'lib', 'libpywrap_tensorflow_internal.so')
        call(['ln', '-sf', src, tgt])
        call(['ln', '-sf', 'libpywrap_tensorflow_internal.so', join(context, 'lib', '_pywrap_tensorflow_internal.so')])

        libs = ('-lpywrap_tensorflow_internal '
                '-ltensorflow_framework')

        # copy headers to the context dir
        include_dir = join(context, 'include', 'tensorflow', 'c')
        rmtree(include_dir, ignore_errors=True)
        copytree(join(dirname(__file__), 'c'), include_dir)

    elif args.mode == 'libtensorflow':
        print('downloading libtensorflow...')
        with_gpu = args.with_gpu
        platform = args.platform
        version = args.version

        hardware = 'cpu'
        if with_gpu:
            hardware = 'gpu'

        tarball = ('libtensorflow-{}-{}-x86_64-{}.tar.gz'.format(hardware, platform, version))
        url = ('https://storage.googleapis.com/tensorflow/libtensorflow/{}'.format(tarball))

        # download the tarball
        call(['wget', url])

        # copy it to the given context
        print('extracting...')
        call(['tar', '-C', context, '-xzf', tarball])
        remove(tarball)

        libs = '-ltensorflow'

    else:
        raise(Exception('Not valid operation mode chosen.'))

    # create the pkg-config file

    print('preparing pkg-config file...')
    includes = '-I' + join(context, 'include/tensorflow')
    lib_dirs = '-L' + join(context, 'lib')

    pkg_config = ('Name: tensorflow\n'
                  'Description: machine learning lib -- development files\n'
                  'Version: {}\n'
                  'Libs: {} {}\n'
                  'Cflags: {}\n').format(version, lib_dirs, libs, includes)

    with open(join(context, 'lib', 'pkgconfig', 'tensorflow.pc'), 'w') as f:
        f.write(pkg_config)

    # sometimes the dynamic linker has to be reconfigured
    print('reconfiguring the linker...')
    call(['ldconfig'])

    print('done!')
