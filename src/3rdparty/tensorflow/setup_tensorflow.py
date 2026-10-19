#!/usr/bin/env python3

# Copyright (C) 2006-2025  Music Technology Group - Universitat Pompeu Fabra
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

"""Generate a tensorflow.pc describing the TensorFlow C API inside a pip package.

Essentia links against the TensorFlow C API and finds it with pkg-config. Some
platforms ship a packaged C library that already provides tensorflow.pc (Homebrew's
libtensorflow, several distributions); where they do not, the pip `tensorflow` wheel
is the only maintained source of the C API, and this script points pkg-config at it.

The wheel exports the C API from libtensorflow_cc, next to libtensorflow_framework,
and carries a matching header tree under tensorflow/include. Those files are used
where they are: this script only writes a handful of symlinks with linker-friendly
names plus the .pc itself, so it never copies the ~600 MB library.

TensorFlow 2.13 is the floor. Earlier wheels exported the C API from the Python
wrapper extension (_pywrap_tensorflow_internal), which is not linkable on its own.
"""

import argparse
import os
import subprocess
import sys
from os.path import abspath, basename, dirname, exists, isdir, join

# Minimum pip TensorFlow whose layout this script understands.
MIN_VERSION = (2, 13)

# Library stems that may export the C API, most specific first. Wheels ship
# libtensorflow_cc; a standalone C library build ships libtensorflow.
C_API_STEMS = ('tensorflow_cc', 'tensorflow')

# Linked alongside the C API library when the package ships it.
SUPPORT_STEMS = ('tensorflow_framework',)

# The header every TensorflowPredict* algorithm includes, relative to an include dir.
C_API_HEADER = join('tensorflow', 'c', 'c_api.h')

PC_TEMPLATE = """prefix={prefix}
libdir={libdir}
includedir={includedir}
package_dir={package_dir}

Name: tensorflow
Description: TensorFlow C API from the pip tensorflow package
Version: {version}
Requires:
Libs: -L${{libdir}} {libs} -Wl,-rpath,{rpath}{rpath_link}
Cflags: -I${{includedir}}
"""

# `package_dir` above is not consumed by the linker. It records where the pip package
# actually lives, so that `pkg-config --variable=package_dir tensorflow` answers the
# question without importing TensorFlow. src/wscript reads it when
# --with-tensorflow-pip-rpath / ESSENTIA_TENSORFLOW_PIP_RPATH asks for a wheel build,
# and the libdir above is no help there: it holds symlinks, not the real libraries.


def die(message):
    """Report a fatal problem and exit non-zero."""
    sys.stderr.write('error: %s\n' % message)
    raise SystemExit(1)


def locate_package(python):
    """Return the directory of the tensorflow package importable by `python`.

    Only stdout is read. Importing TensorFlow writes an unconditional line to
    stderr -- "Could not find cuda drivers" on linux, "This TensorFlow binary is
    optimized to use available CPU instructions" on macOS -- and folding that
    into stdout puts a log line in the middle of the path. stderr is still
    captured, but separately, so a failing import can still be reported in full.
    """
    code = 'import os.path, tensorflow; print(os.path.dirname(tensorflow.__file__))'
    try:
        process = subprocess.Popen([python, '-c', code],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
    except OSError as error:
        die('could not run %s: %s' % (python, error))
    if process.returncode != 0:
        detail = err.decode('utf-8', 'replace').strip()
        die('%s cannot import tensorflow.\n\n%s\n\n'
            'Install it (`%s -m pip install "tensorflow>=%d.%d"`), or pass --package-dir '
            'to point at an unpacked wheel.'
            % (python, detail, python, MIN_VERSION[0], MIN_VERSION[1]))
    return out.decode('utf-8', 'replace').strip()


def read_version(package_dir):
    """Read the TensorFlow version from the dist-info directory beside the package.

    Reading the metadata rather than importing keeps this usable against an unpacked
    wheel, including one built for a different Python version or platform than the
    interpreter running this script.
    """
    parent = dirname(package_dir)
    for entry in sorted(os.listdir(parent)):
        if entry.startswith('tensorflow') and entry.endswith('.dist-info'):
            name = entry[:-len('.dist-info')]
            if '-' in name:
                return name.rsplit('-', 1)[1]
    return None


def check_version(version):
    """Reject TensorFlow releases whose layout this script does not understand."""
    if version is None:
        sys.stderr.write('warning: could not determine the TensorFlow version; '
                         'assuming it is at least %d.%d\n' % MIN_VERSION)
        return '%d.%d' % MIN_VERSION

    try:
        numbers = tuple(int(part) for part in version.split('.')[:2])
    except ValueError:
        sys.stderr.write('warning: unparseable TensorFlow version %r; continuing\n' % version)
        return version

    if numbers < MIN_VERSION:
        die('found TensorFlow %s, but this script needs >= %d.%d.\n'
            'Older wheels export the C API from the Python wrapper extension, which '
            'cannot be linked against.' % (version, MIN_VERSION[0], MIN_VERSION[1]))
    return version


def find_libraries(package_dir):
    """Return (stem, filename) for each library to link, C API first.

    Wheels ship versioned filenames only (libtensorflow_cc.2.dylib,
    libtensorflow_cc.so.2), which no linker will find from -ltensorflow_cc.
    """
    found = {}
    for name in sorted(os.listdir(package_dir)):
        for stem in C_API_STEMS + SUPPORT_STEMS:
            if name.startswith('lib%s.' % stem) and ('.so' in name or '.dylib' in name):
                found.setdefault(stem, name)

    c_api = [stem for stem in C_API_STEMS if stem in found]
    if not c_api:
        die('no TensorFlow C API library in %s.\nExpected one of: %s.'
            % (package_dir, ', '.join('lib%s' % stem for stem in C_API_STEMS)))

    stems = [c_api[0]] + [stem for stem in SUPPORT_STEMS if stem in found]
    return [(stem, found[stem]) for stem in stems]


def find_rpath_link_dirs(package_dir):
    """Return directories ld needs on its *link-time* search path, as -rpath-link flags.

    auditwheel builds the linux tensorflow wheel by moving the libraries it vendored into
    a sibling `<distribution>.libs` directory and pointing libtensorflow_cc's RPATH at it.
    That is enough at run time, but not when ld links an executable against
    libtensorflow_cc: ld resolves DT_NEEDED transitively and stops with
    "undefined reference to `__kmpc_fork_call'" unless it can find libomp-<hash>.so.5.
    -rpath-link puts the directory on ld's search path without recording it in the binary,
    which is what we want -- at run time libtensorflow_cc finds its own dependencies.

    Empty on macOS, where the dylibs have no such sibling directory and ld64 has no
    -rpath-link.
    """
    if sys.platform == 'darwin':
        return []

    parent = dirname(package_dir)
    prefix = basename(package_dir)
    dirs = []
    for entry in sorted(os.listdir(parent)):
        if (entry.startswith(prefix) and entry.endswith('.libs')
                and isdir(join(parent, entry))):
            dirs.append(join(parent, entry))
    return dirs


def find_includedir(package_dir):
    """Return the include dir from which <tensorflow/c/c_api.h> resolves."""
    includedir = join(package_dir, 'include')
    if not exists(join(includedir, C_API_HEADER)):
        die('no C API headers in %s (expected %s).\n'
            'This does not look like a complete TensorFlow wheel.'
            % (includedir, C_API_HEADER))
    return includedir


def link(source, target):
    """Create or replace a symlink at `target` pointing to `source`."""
    if os.path.islink(target) or exists(target):
        os.remove(target)
    os.symlink(source, target)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='After running this, add <prefix>/lib/pkgconfig to PKG_CONFIG_PATH and\n'
               'configure Essentia with `./waf configure --with-tensorflow`.')
    parser.add_argument('--prefix', '-p', default='/usr/local',
                        help='where to write lib/pkgconfig/tensorflow.pc and the library '
                             'symlinks (default: %(default)s; must be writable)')
    parser.add_argument('--package-dir', '-d', default=None,
                        help='use this tensorflow package directory instead of importing '
                             'tensorflow; accepts the tensorflow/ directory of an '
                             'unpacked wheel')
    parser.add_argument('--python', default=sys.executable,
                        help='interpreter used to locate the tensorflow package '
                             '(default: %(default)s)')
    args = parser.parse_args()

    if args.package_dir:
        package_dir = abspath(args.package_dir)
        if not isdir(package_dir):
            die('%s is not a directory' % package_dir)
    else:
        package_dir = locate_package(args.python)

    print('using the tensorflow package in %s' % package_dir)

    version = check_version(read_version(package_dir))
    print('tensorflow version: %s' % version)

    libraries = find_libraries(package_dir)
    includedir = find_includedir(package_dir)
    rpath_link_dirs = find_rpath_link_dirs(package_dir)

    libdir = join(abspath(args.prefix), 'lib')
    pkgconfig_dir = join(libdir, 'pkgconfig')
    try:
        os.makedirs(pkgconfig_dir)
    except OSError:
        if not isdir(pkgconfig_dir):
            raise

    # Give each versioned library a name the linker accepts from -l<stem>. The loader
    # still resolves the versioned SONAME/install_name at run time, which is what the
    # -Wl,-rpath in the .pc points at.
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    for stem, filename in libraries:
        target = join(libdir, 'lib%s%s' % (stem, suffix))
        link(join(package_dir, filename), target)
        print('%s -> %s' % (target, filename))

    pkg_config = PC_TEMPLATE.format(
        prefix=abspath(args.prefix),
        libdir=libdir,
        includedir=includedir,
        version=version,
        libs=' '.join('-l%s' % stem for stem, _ in libraries),
        rpath=package_dir,
        rpath_link=''.join(' -Wl,-rpath-link,%s' % path for path in rpath_link_dirs),
        package_dir=package_dir)

    path = join(pkgconfig_dir, 'tensorflow.pc')
    with open(path, 'w') as pcfile:
        pcfile.write(pkg_config)

    print('\nwrote %s:\n' % path)
    print(pkg_config)
    print('add %s to PKG_CONFIG_PATH, then run:\n'
          '    ./waf configure --with-tensorflow' % pkgconfig_dir)


if __name__ == '__main__':
    main()
