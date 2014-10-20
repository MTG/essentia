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



def configure(ctx):
    print('→ configuring the project in ' + ctx.path.abspath())

    ctx.env.VERSION = VERSION
    ctx.env.GIT_SHA = GIT_SHA

    ctx.env.WITH_CPPTESTS = ctx.options.WITH_CPPTESTS

    # compiler flags
    ctx.env.CXXFLAGS = [ '-pipe', '-Wall' ]

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

    # global defines
    ctx.env.DEFINES = []

    ctx.load('compiler_cxx compiler_c')

    if ctx.env.DEST_OS != 'win32':
        # required if we want to use libessentia.a to be linked in the python bindings
        # (dynamic library, needs -fPIC)
        # all code is position independent on windows, so don't include it there
        ctx.env.CXXFLAGS += [ '-fPIC' ]

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
        # missing -lpthread flag on Ubuntu
        if platform.dist()[0] == 'Ubuntu':
            ext_paths = ['/usr/lib/i386-linux-gnu', '/usr/lib/x86_64-linux-gnu']
            ctx.read_shlib('pthread', paths=ext_paths)
            ctx.env.USES += ' pthread'

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
