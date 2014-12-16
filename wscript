#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import platform

def get_git_version():
    """ try grab the current version number from git"""
    version = "Undefined"
    if os.path.exists(".git"):
        try:
            version = os.popen("git describe --dirty --always").read().strip()
        except Exception, e:
            print e
    return version


APPNAME = 'essentia'
VERSION = open('VERSION', 'r').read().strip('\n')
GIT_SHA = get_git_version();

top = '.'
out = 'build'


def options(ctx):
    ctx.load('compiler_cxx compiler_c')
    ctx.recurse('src')

    ctx.add_option('--with-cpptests', action='store_true',
                   dest='WITH_CPPTESTS', default=False,
                   help='build the c++ tests')

    ctx.add_option('--mode', action='store',
                   dest='MODE', default="release",
                   help='debug or release')

    ctx.add_option('--cross-compile-mingw32', action='store_true',
                   dest='CROSS_COMPILE_MINGW32', default=False,
                   help='cross-compile for windows using mingw32 on linux')


def configure(ctx):
    print('→ configuring the project in ' + ctx.path.abspath())

    ctx.env.VERSION = VERSION
    ctx.env.GIT_SHA = GIT_SHA

    ctx.env.WITH_CPPTESTS = ctx.options.WITH_CPPTESTS

    # compiler flags
    ctx.env.CXXFLAGS = [ '-pipe', '-Wall' ]

    # force using SSE floating point (default for 64bit in gcc) instead of
    # 387 floating point (used for 32bit in gcc) to avoid numerical differences
    # between 32 and 64bit builds (see https://github.com/MTG/essentia/issues/179)
    ctx.env.CXXFLAGS += [ '-msse', '-msse2', '-mfpmath=sse' ]

    # define this to be stricter, but sometimes some libraries can give problems...
    #ctx.env.CXXFLAGS += [ '-Werror' ]

    if ctx.options.MODE == 'debug':
        print ('→ Building in debug mode')
        ctx.env.CXXFLAGS += [ '-g' ]

    elif ctx.options.MODE == 'release':
        print ('→ Building in release mode')
        ctx.env.CXXFLAGS += [ '-O2' ] # '-march=native' ] # '-msse3', '-mfpmath=sse' ]

    else:
        raise ValueError('mode should be either "debug" or "release"')

    if not ctx.options.CROSS_COMPILE_MINGW32 and sys.platform != 'win32':
        # required if we want to use libessentia.a to be linked in the python bindings
        # (dynamic library, needs -fPIC)
        ctx.env.CXXFLAGS += [ '-fPIC' ]

    # global defines
    ctx.env.DEFINES = []

    if sys.platform == 'darwin':
        # clang fails on 10.7 using <atomic>, because libc++ is not new enough
        #ctx.env.CC = 'clang'
        #ctx.env.CXX = 'clang++'
        ctx.env.CXXFLAGS = [ '-stdlib=libc++', '-std=c++11', '-Wno-gnu' ]
        #ctx.env.LINKFLAGS = [ '-stdlib=libc++' ]

        ctx.env.DEFINES   += [ 'GTEST_HAS_TR1_TUPLE=0' ]
        # for defining static const variables in header
        # ctx.env.CXXFLAGS += [ '-Wno-static-float-init' ]
        # add /usr/local/include as the brew formula for yaml doesn't have
        # the cflags properly set
        #ctx.env.CXXFLAGS += [ '-I/usr/local/include' ]

    elif sys.platform.startswith('linux'):
        # include -pthread flag because not all versions of gcc provide it automatically
        ctx.env.CXXFLAGS += [ '-pthread' ]

    #elif sys.platform == 'win32':
    #    # compile libgcc and libstd statically when using MinGW
    #    ctx.env.CXXFLAGS = [ '-static-libgcc', '-static-libstdc++' ]

    #    # make pkgconfig find 3rdparty libraries in packaging/win32_3rdparty
    #    os.environ["PKG_CONFIG_PATH"] = 'packaging\win32_3rdparty\lib\pkgconfig'
    #    os.environ["PKG_CONFIG_LIBDIR"] = os.environ["PKG_CONFIG_PATH"]
    #
    #    # TODO why this code does not work?
    #    # force the use of mingw gcc compiler instead of msvc
    #    #ctx.env.CC = 'gcc'
    #    #ctx.env.CXX = 'g++'


    # use manually prebuilt dependencies in the case of static examples or mingw cross-build
    if ctx.options.CROSS_COMPILE_MINGW32:
        # locate mingw32 compilers and use them
        ctx.find_program('i686-w64-mingw32-gcc', var='CC')
        ctx.find_program('i686-w64-mingw32-g++', var='CXX')
        ctx.find_program('i686-w64-mingw32-ar', var='AR')

        # compile libgcc and libstd statically when using MinGW
        ctx.env.CXXFLAGS = [ '-static-libgcc', '-static-libstdc++' ]

        print ("→ Cross-compiling with MinGW32: search for pre-built dependencies in 'packaging/win32_3rdparty'")
        os.environ["PKG_CONFIG_PATH"] = 'packaging/win32_3rdparty/lib/pkgconfig'
        os.environ["PKG_CONFIG_LIBDIR"] = os.environ["PKG_CONFIG_PATH"]

    elif ctx.options.WITH_STATIC_EXAMPLES and (sys.platform.startswith('linux') or sys.platform == 'darwin'):
        print ("→ Compiling with static examples on Linux: search for pre-built dependencies in 'packaging/debian'")
        os.environ["PKG_CONFIG_PATH"] = 'packaging/debian_3rdparty/lib/pkgconfig'
        os.environ["PKG_CONFIG_LIBDIR"] = os.environ["PKG_CONFIG_PATH"]

    ctx.load('compiler_cxx compiler_c')

    # write pkg-config file
    prefix = os.path.normpath(ctx.options.prefix)
    opts = { 'prefix': prefix,
             'version': ctx.env.VERSION,
             }

    pcfile = '''prefix=%(prefix)s
    libdir=${prefix}/lib
    includedir=${prefix}/include

    Name: libessentia
    Description: audio analysis library -- development files
    Version: %(version)s
    Libs: -L${libdir} -lfftw3 -lyaml -lavcodec -lavformat -lavutil -lsamplerate -ltag -lfftw3f -lgaia2
    Cflags: -I${includedir}/essentia I${includedir}/essentia/scheduler I${includedir}/essentia/streaming I${includedir}/essentia/utils
    ''' % opts

    pcfile = '\n'.join([ l.strip() for l in pcfile.split('\n') ])
    ctx.env.pcfile = pcfile
    #open('build/essentia.pc', 'w').write(pcfile) # we'll do it later on the build stage

    ctx.recurse('src')


def adjust(objs, path):
    return [ '%s/%s' % (path, obj) for obj in objs ]

def build(ctx):
    print('→ building from ' + ctx.path.abspath())

    ctx.recurse('src')

    if ctx.env.WITH_CPPTESTS:
        ctx.program(
            source   = ctx.path.ant_glob('test/src/basetest/*.cpp test/3rdparty/gtest-1.6.0/src/gtest-all.cc '),
            target   = 'basetest',
            includes = [ 'test/3rdparty/gtest-1.6.0/include',
                         'test/3rdparty/gtest-1.6.0' ] + adjust(ctx.env.INCLUDES, 'src'),
            install_path = None,
            use      = 'essentia ' + ctx.env.USES
            )

def run_tests(ctx):
    os.system(out + '/basetest')

def run_python_tests(ctx):
    os.system('python test/src/unittest/all_tests.py')

def ipython(ctx):
    os.system('ipython --pylab')

def doc(ctx):
    os.system('doc/build_sphinx_doc.sh')
