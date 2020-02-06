#!/usr/bin/env python

# Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program.  If not, see http://www.gnu.org/licenses/


import argparse
from os import symlink, remove
from os.path import join, dirname, abspath
from shutil import copytree, rmtree
from subprocess import call


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Sets up Tensorflow for linking against it.'
        'This can be done in two ways:'
        ' - python: By symlinking to an existing Tensorflow Python package. '
        '     This is the recommended way if Tensorflow is avaialble from Python.'
        ' - libtensorflow: By downloading and installing the C API. This mode does not '
        '     allow simultaneous use of Essentia and Tensorflow from Python due to symbol name conflicts.')

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
        # WARNING. With `--mode libtensorflow`, the following problem is known
        # to arise when importing Essentia and Tensorflow in Python at the same time.
        #   In [1]: import tensorflow
        #   In [2]: import essentia
        #
        #   ImportError: /usr/local/lib/libtensorflow.so.1: undefined symbol:
        #  _ZN6google8protobuf5Arena18CreateMaybeMessageIN10tensorflow16OptimizerOptionsEIEEEPT_PS1_DpOT0_

        # The recommended solution is to link Essentia against the Tensorflow shared
        # libraries shipped with the Tensorflow wheel with `--mode python`

        print('downloading libtensorflow...')
        with_gpu = args.with_gpu
        platform = args.platform
        version = args.version

        hardware = 'gpu' if with_gpu else 'cpu'

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
