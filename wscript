#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import platform

def get_git_version():
    """ try grab the current version number from git"""
    version = "Undefined"
    if os.path.exists(".git"):
        try:
            version = os.popen("git describe --dirty --always").read().strip()
        except Exception as e:
            print(e)
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

    ctx.add_option('--arch', action='store',
                   dest='ARCH', default="x64",
                   help='Target architecture when compiling on OSX: i386, x64 or FAT')

    ctx.add_option('--cross-compile-mingw32', action='store_true',
                   dest='CROSS_COMPILE_MINGW32', default=False,
                   help='cross-compile for windows using mingw32 on linux')

    ctx.add_option('--cross-compile-android', action='store_true',
                   dest='CROSS_COMPILE_ANDROID', default=False,
                   help='cross-compile for Android using toolchain')

    ctx.add_option('--cross-compile-ios', action='store_true',
                   dest='CROSS_COMPILE_IOS', default=False,
                   help='cross-compile for iOS (ARMv7 and ARM64)')

    ctx.add_option('--cross-compile-ios-sim', action='store_true',
                   dest='CROSS_COMPILE_IOS_SIM', default=False,
                   help='cross-compile for iOS (i386)')

    ctx.add_option('--emscripten', action='store_true',
                   dest='EMSCRIPTEN', default=False,
                   help='compile Essentia to Javascript with Emscripten')


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
    if not ctx.options.EMSCRIPTEN and not ctx.options.CROSS_COMPILE_ANDROID and not ctx.options.CROSS_COMPILE_IOS:
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
        ctx.env.CPPFLAGS += [ '-fPIC' ] # need that for KissFFT


    ctx.env.CROSS_COMPILE_MINGW32 = ctx.options.CROSS_COMPILE_MINGW32

    # global defines
    ctx.env.DEFINES = []

    if ctx.options.EMSCRIPTEN:
        ctx.env.CXXFLAGS += [ '-I' + os.path.join(os.environ['EMSCRIPTEN'], 'system', 'lib', 'libcxxabi', 'include') ]
        ctx.env.CXXFLAGS += ['-Oz']
    elif sys.platform == 'darwin':
        # clang fails on 10.7 using <atomic>, because libc++ is not new enough
        #ctx.env.CC = 'clang'
        #ctx.env.CXX = 'clang++'
        #ctx.env.CXXFLAGS = [ '-stdlib=libc++', '-std=c++11', '-Wno-gnu' ]
        #ctx.env.LINKFLAGS = [ '-stdlib=libc++' ]

        ctx.env.DEFINES   += [ 'GTEST_HAS_TR1_TUPLE=0' ]
        # for defining static const variables in header
        # ctx.env.CXXFLAGS += [ '-Wno-static-float-init' ]
        # add /usr/local/include as the brew formula for yaml doesn't have
        # the cflags properly set
        #ctx.env.CXXFLAGS += [ '-I/usr/local/include' ]

        if ctx.options.ARCH == 'i386':
            ctx.env.CXXFLAGS += [ '-arch' , 'i386']
            ctx.env.LINKFLAGS += [ '-arch', 'i386']
            ctx.env.LDFLAGS = ['-arch', 'i386']
        if ctx.options.ARCH == 'FAT':
            ctx.env.CXXFLAGS += [ '-arch' , 'i386', '-arch', 'x86_64']
            ctx.env.LINKFLAGS += [ '-arch' , 'i386', '-arch', 'x86_64']
            ctx.env.LDFLAGS = [ '-arch' , 'i386', '-arch', 'x86_64']

    elif sys.platform.startswith('linux'):
        # include -pthread flag because not all versions of gcc provide it automatically
        ctx.env.CXXFLAGS += [ '-pthread' ]


    elif sys.platform == 'win32':
        # compile libgcc and libstd statically when using MinGW
        ctx.env.CXXFLAGS = [ '-static-libgcc', '-static-libstdc++' ]

        win_path = "packaging/win32_3rdparty"

        # Establish MINGW locations
        tdm_root = ctx.options.prefix
        tdm_bin = tdm_root + "/bin"
        tdm_include = tdm_root + "/include"
        tdm_lib = tdm_root + "/lib"

        # make pkgconfig find 3rdparty libraries in packaging/win32_3rdparty
        # libs_3rdparty = ['yaml-0.1.5', 'fftw-3.3.3', 'libav-0.8.9', 'libsamplerate-0.1.8']
        # libs_paths = [';packaging\win32_3rdparty\\' + lib + '\lib\pkgconfig' for lib in libs_3rdparty]
        # os.environ["PKG_CONFIG_PATH"] = ';'.join(libs_paths)

        os.environ["PKG_CONFIG_PATH"] = tdm_root + '\lib\pkgconfig'

        # TODO why this code does not work?
        # force the use of mingw gcc compiler instead of msvc
        #ctx.env.CC = 'gcc'
        #ctx.env.CXX = 'g++'

        import distutils.dir_util

        print("copying pkgconfig ...")
        distutils.dir_util.copy_tree(win_path + "/pkgconfig/bin", tdm_bin)

        libs_3rdparty = ['yaml-0.1.5', 'fftw-3.3.3', 'libav-0.8.9', 'libsamplerate-0.1.8', 'taglib-1.9.1']
        for lib in libs_3rdparty:
            print("copying " + lib + "...")
            distutils.dir_util.copy_tree(win_path + "/" + lib + "/include", tdm_include)
            distutils.dir_util.copy_tree(win_path + "/" + lib + "/lib", tdm_lib)

    if ctx.options.CROSS_COMPILE_ANDROID:
        print ("→ Cross-compiling for Android ARM")
        ctx.find_program('arm-linux-androideabi-gcc', var='CC')
        ctx.find_program('arm-linux-androideabi-g++', var='CXX')
        ctx.find_program('arm-linux-androideabi-ar', var='AR')
        ctx.env.LINKFLAGS += ['-Wl,-soname,libessentia.so']

    if ctx.options.CROSS_COMPILE_IOS:
        print ("→ Cross-compiling for iOS (ARMv7 and ARM64)")
        ctx.env.CXXFLAGS += [ '-arch' , 'armv7']
        ctx.env.LINKFLAGS += [ '-arch', 'armv7']
        ctx.env.LDFLAGS += ['-arch', 'armv7']
        ctx.env.CXXFLAGS += [ '-arch' , 'arm64']
        ctx.env.LINKFLAGS += [ '-arch', 'arm64']
        ctx.env.LDFLAGS += ['-arch', 'armv64']

        ctx.env.CXXFLAGS += ['-stdlib=libc++']
        ctx.env.CXXFLAGS += ['-miphoneos-version-min=5.0']
        ctx.env.CXXFLAGS += [ '-isysroot' , '/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk']
        ctx.env.CXXFLAGS += [ '-fembed-bitcode']

    if ctx.options.CROSS_COMPILE_IOS_SIM:
        print ("→ Cross-compiling for iOS Simulator (i386)")
        ctx.env.CXXFLAGS += [ '-arch' , 'i386']
        ctx.env.LINKFLAGS += [ '-arch', 'i386']
        ctx.env.LDFLAGS += ['-arch', 'i386']
        ctx.env.CXXFLAGS += [ '-arch' , 'x86_64']
        ctx.env.LINKFLAGS += [ '-arch', 'x86_64']
        ctx.env.LDFLAGS += ['-arch', 'x86_64']

        ctx.env.CXXFLAGS += ['-stdlib=libc++']
        ctx.env.CXXFLAGS += ['-miphoneos-version-min=5.0']
        ctx.env.CXXFLAGS += [ '-isysroot' , '/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk']

    # use manually prebuilt dependencies in the case of static examples or mingw cross-build
    if ctx.options.CROSS_COMPILE_MINGW32:
        print ("→ Cross-compiling for Windows with MinGW: search for pre-built dependencies in 'packaging/win32_3rdparty'")
        os.environ["PKG_CONFIG_PATH"] = 'packaging/win32_3rdparty/lib/pkgconfig'
        os.environ["PKG_CONFIG_LIBDIR"] = os.environ["PKG_CONFIG_PATH"]

        # locate MinGW compilers and use them
        ctx.find_program('i686-w64-mingw32-gcc', var='CC')
        ctx.find_program('i686-w64-mingw32-g++', var='CXX')
        ctx.find_program('i686-w64-mingw32-ar', var='AR')

        # compile libgcc and libstd statically when using MinGW
        ctx.env.CXXFLAGS = [ '-static-libgcc', '-static-libstdc++' ]

    elif ctx.options.WITH_STATIC_EXAMPLES and (sys.platform.startswith('linux') or sys.platform == 'darwin'):
        print ("→ Compiling with static examples on Linux/OSX: search for pre-built dependencies in 'packaging/debian'")
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
    Libs: -L${libdir} -lessentia -lgaia2 -lfftw3 -lyaml -lavcodec -lavformat -lavutil -lavresample -lsamplerate -ltag -lfftw3f
    Cflags: -I${includedir}/essentia -I${includedir}/essentia/scheduler -I${includedir}/essentia/streaming -I${includedir}/essentia/utils
    ''' % opts

    pcfile = '\n'.join([ l.strip() for l in pcfile.split('\n') ])
    ctx.env.pcfile = pcfile
    #open('build/essentia.pc', 'w').write(pcfile) # we'll do it later on the build stage

    ctx.recurse('src')


def adjust(objs, path):
    return [ '%s/%s' % (path, obj) for obj in objs ]

def build(ctx):
    print('→ building from ' + ctx.path.abspath())

    # missing -lpthread flag on Ubuntu
    if platform.dist()[0] == 'Ubuntu' and not ctx.env.CROSS_COMPILE_MINGW32 and not ctx.env.WITH_STATIC_EXAMPLES:
        ext_paths = ['/usr/lib/i386-linux-gnu', '/usr/lib/x86_64-linux-gnu']
        ctx.read_shlib('pthread', paths=ext_paths)
        ctx.env.USES += ' pthread'

    ctx.recurse('src')

    if ctx.env.WITH_CPPTESTS:
        # missing -lpthread flag on Ubuntu and LinuxMint
        if platform.dist()[0] in ['Ubuntu', 'LinuxMint']:
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
    ret = os.system(out + '/basetest')
    if ret:
        ctx.fatal('failed to run tests. Check test output')

def run_python_tests(ctx):
    # create a local python package folder
    os.system('mkdir -p build/python')
    os.system('cp -r src/python/essentia build/python/')
    os.system('cp build/src/python/_essentia.so build/python/essentia')

    ret = os.system('PYTHONPATH=build/python python test/src/unittest/all_tests.py')
    if ret:
        ctx.fatal('failed to run python tests. Check test output')

def ipython(ctx):
    os.system('ipython --pylab')

def doc(ctx):
    # create a local python package folder
    os.system('mkdir -p build/python')
    os.system('cp -r src/python/essentia build/python/essentia')
    os.system('cp build/src/python/_essentia.so build/python')
    pythonpath = os.path.abspath('build/python')

    os.system('PYTHONPATH=%s doc/build_sphinx_doc.sh' % pythonpath)
