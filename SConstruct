#!/usr/bin/env python

# Main SConstruct file

# import environment variables
execfile('SConstruct_config')
source_tree_path = os.getcwd()
Export('env', 'conf', 'source_tree_path')
set_root(env, source_tree_path)

# ran_scripts is a list of paths (relative to source_tree_path) that have already been run. It is
# used to prevent the same script to be run twice. All subsidary SConscripts should make use of it.
ran_scripts = []
Export('ran_scripts')

Help("""
The following building commands are available:
  'scons' to build the library
  'scons install' to install the library
  'scons python' to build the python module
  'scons python_install' to install the python module
  'scons tests' to build the C++ tests
  'scons run_tests' to run the C++ tests
  'scons run_python_tests' to run the non-streaming unittests
  'scons examples' to build the examples
  'scons static_examples' to build the examples statically (only on linux)
  'scons run_examples' to run examples
  'scons vamp' to build the Vamp plugin wrappers
  'scons python_extractor' to generate cpp streaming_extractor from python
  'scons doc' generate algorithm, descriptor, dependency and doxygen documentation
  'scons check_filenames' to check the consistency of algorithms source files naming

When used with '-c', any of the above commands (e.g. 'scons -c test'), will
clean the files produced by that command and all of its dependencies.

Local compilation flags:
   prefix                 installation prefix (default: None)
   destdir                installation directory (default: /usr/local)
   thirdparty             thirdparty libs directory (default: None)
   ffmpeg_dir             ffmpeg's installation directory (default: None)
   algoignore             specify algorithms to be excluded when building (default: None)
   algoinclude            specify algorithms to be included when building (default: None)
   streaming              set to true to build streaming mode (default: True)
   mode                   set to release/debug/optimized (default: release)
   crosscompileppc        set to true for ppc cross compiling (default: False)
   icc                    set to true for intel compiler (default: False)
   use_ffmpeg             set to true to build with ffmpeg (default: True)
   use_libsamplerate      set to true to build with libsamplerate (default: True)
   use_taglib             set to true to build with taglib (default: True)
   use_pthread            set to true to build with pthread support (default: False)
   use_tbb                set to true to build with tbb support (default: False)
   use_encryption         set to true to encrypt yaml output (default: False)
   use_sdk10.4            set to true for Mac OS X 10.4 SDK support (default: False)
   use_gaia               set to true to build with Gaia plugin support (default: False)
   win64                  set to true for 64bit support on Windows (default: False)
   strip_documentation    set to true to false to have documentation available (default: False)
   info                   set to true to have algorithm info available (default: False)
   prof_genx              set to true to generate profiling information (default: False)

""")

### default action is to build ###
if not COMMAND_LINE_TARGETS: COMMAND_LINE_TARGETS.append('build')

### Process algoignore/include arguments ###
from algorithms_info import get_all_algorithms
algorithms = get_all_algorithms(join('src','algorithms'))

algo_ignore_arg = ARGUMENTS.get('algoignore', None)
if conf.ALGOIGNORE: algo_ignore = conf.ALGOIGNORE
else: algo_ignore = []
if algo_ignore_arg:
    for name in algo_ignore_arg.split(","):
        if not name in algorithms.keys():
            print('Trying to ignore non-existent algorithm: '+name)
            Exit(1)
        algo_ignore.append(name)

algo_include_arg = ARGUMENTS.get('algoinclude', None)
algo_include = []
if algo_include_arg:
    for name in algo_include_arg.split(","):
        if not name in algorithms.keys():
            print('Trying to include non-existent algorithm: '+name)
            Exit(1)
        algo_include.append(name)

# print a confirmation message
if algo_ignore or algo_include:
    print('-'*80)
    print('Ignoring the following algorithms: '+(' '.join(algo_ignore)))
    print('Including only the following algorithms: '+(' '.join(algo_include)))
    print('-'*80)

for algoname in algorithms.keys():
    if (algo_include and not algoname in algo_include) or \
       not algo_include and algoname in algo_ignore:
        del algorithms[algoname]

Export('algorithms')


### Dispatch on target ###
cmds = COMMAND_LINE_TARGETS
if 'build'            in cmds: SConscript(join('src','SConscript'))
elif 'install'          in cmds: SConscript(join('src','SConscript'))
elif 'python'           in cmds: SConscript(join('src','python','SConscript'))
elif 'python_install'   in cmds: SConscript(join('src','python','SConscript'))
elif 'examples'         in cmds: SConscript(join('src','examples','SConscript'))
elif 'static_examples'  in cmds: SConscript(join('src','examples','SConscript'))
elif 'run_examples'     in cmds: SConscript(join('src','examples','SConscript'))
elif 'vamp'             in cmds: SConscript(join('src','examples','SConscript'))
elif 'python_extractor' in cmds: SConscript(join('src','examples', 'python', 'streaming_extractor', 'SConscript'))
elif 'tests'            in cmds: SConscript(join('test','src','basetest','SConscript'))
elif 'run_tests'        in cmds: SConscript(join('test','src','basetest','SConscript'))
elif 'run_python_tests' in cmds: SConscript(join('test','src','unittest','SConscript'))
elif 'doc'              in cmds: SConscript(join('doc','SConscript'))
elif 'package'          in cmds: SConscript(join('packaging','SConscript'))

else: print('Please supply a valid target to build')



pcTemplate = '''prefix=%s
exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir=${prefix}/include/essentia

Name: Essentia
Description: A library for content analysis of audio files
Version: %s
Libs: -L${libdir} -lessentia -lfftw3f -lsamplerate -lavcodec -lavformat -lavutil -ltag -ltbb -lyaml
Cflags: -I${includedir}
'''

# create & install the pkg-config file if necessary
commands = COMMAND_LINE_TARGETS
if any_of(commands, ['install', 'package']) and (sys.platform in [ 'linux2', 'darwin' ]):
    pcfiledir = join(conf.DESTDIR, conf.PREFIX, 'lib', 'pkgconfig')
    #pcfiledir = join(ARGUMENTS.get('destdir',''), conf.PREFIX, 'lib', 'pkgconfig')
    if 'package' in commands: pcfiledir = join(source_tree_path, 'build', 'packaging', 'lib', 'pkgconfig')
    pccontents = pcTemplate % (join('/',conf.PREFIX), open('VERSION').read())
    try:
        os.system('mkdir -p ' + pcfiledir)
        open(pcfiledir + '/essentia.pc', 'w').write(pccontents)
    except IOError:
        print 'ERROR: you need to have super user privileges in order to be able to install Essentia in ' + pcfiledir
        sys.exit(1)

#if any_of(commands, ['build', 'install', 'package']):

#    if sys.platform in [ 'win32', 'cygwin' ]:
#        env.Tool('vsprojscons', toolpath=['scons'])

#        vcProjectFile = env.File("./essentiaVC80.vcproj")
#        env.Precious( vcProjectFile ) # don't delete on rebuild
#        env.NoClean( vcProjectFile ) # don't delete on clean
#        vcProj = env.SyncVCProj( vcProjectFile, all_objects )

#        env.Depends(vcProj, 'SConstruct_environment')
#        env.Depends(vcProj, 'SConstruct')
#        env.Depends(vcProj, 'src/SConscript')
#        env.Depends(vcProj, 'src/algorithms/SConscript')
#        env.Depends(vcProj, 'src/base/SConscript')
#        env.Depends(vcProj, 'scons/vsprojscons.py')
#        env.Depends(vcProj, 'scons/vsprojsync.py')

#    if sys.platform in [ 'linux2', 'win32', 'cygwin', 'darwin' ]:
#        LIBRARY_PATH = join(conf.DESTDIR, conf.PREFIX, 'lib')

#        # install library
#        lib_install_alias = env.Alias('install', LIBRARY_PATH)
#        Export('lib_install_alias')
#        env.Install(LIBRARY_PATH, lib)
