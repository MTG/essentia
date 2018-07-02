import shutil
import os
import glob
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib

library = None
PYTHON = sys.executable

class EssentiaInstall(install_lib):
    def install(self):
        global library
        install_dir = os.path.join(self.install_dir, library.split(os.sep)[-1])
        res = shutil.move(library, install_dir)
        os.system("ls -l %s" % self.install_dir)
        return [install_dir]


class EssentiaBuildExtension(build_ext):
    def run(self):
        global library
        os.system('rm -rf tmp; mkdir tmp')

        # Ugly hack using an enviroment variable... There's no way to pass a
        # custom flag to python setup.py bdist_wheel
        var_skip_3rdparty = 'ESSENTIA_WHEEL_SKIP_3RDPARTY'
        var_only_python = 'ESSENTIA_WHEEL_ONLY_PYTHON'

        if var_skip_3rdparty in os.environ and os.environ[var_skip_3rdparty]=='1':
            print('Skipping building static 3rdparty dependencies (%s=1)' %  var_skip_3rdparty)
        else:
            os.system('./packaging/build_3rdparty_static_debian.sh')

        if var_only_python in os.environ and os.environ[var_only_python]=='1':
            print('Skipping building the core libessentia library (%s=1)' %  var_only_python)
            os.system('%s waf configure --only-python --static-dependencies '
                      '--prefix=tmp' % PYTHON)
        else:
            os.system('%s waf configure --build-static --static-dependencies '
                      '--with-python --prefix=tmp' % PYTHON)
        os.system('%s waf' % PYTHON)
        os.system('%s waf install' % PYTHON)

        library = glob.glob('tmp/lib/python*/*-packages/essentia')[0]


def get_git_version():
    """ try grab the current version number from git"""
    version = None
    if os.path.exists(".git"):
        try:
            version = os.popen("git describe --always").read().strip()
        except Exception as e:
            print(e)
    return version


def get_version():
    version = open('VERSION', 'r').read().strip('\n')
    if version.count('-dev'):
        # Development version. Get the number of commits after the last release
        git_version = get_git_version()
        dev_commits = git_version.split('-')[-2] if git_version else ''
        if not dev_commits.isdigit():
            print('Error parsing the number of dev commits: %s', dev_commits)
            dev_commits = '0'
        version += dev_commits
    return version


classifiers = [
    'License :: OSI Approved :: GNU Affero General Public License v3',
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Libraries',
    'Topic :: Multimedia :: Sound/Audio :: Analysis',
    'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis',
    'Operating System :: POSIX',
    'Operating System :: MacOS :: MacOS X',
    #'Operating System :: Microsoft :: Windows',
    'Programming Language :: C++',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
]


build_requires = ['numpy>=1.8.2', 'six', 'pyyaml']

module = Extension('name', sources=[])

setup(
    name='essentia',
    version=get_version(),
    description='Library for audio and music analysis, description and synthesis',
    long_description='C++ library for audio and music analysis, description and synthesis, including Python bindings',
    author='Dmitry Bogdanov',
    author_email='dmitry.bogdanov@upf.edu',
    url='http://essentia.upf.edu',
    project_urls={
        "Documentation": "http://essentia.upf.edu",
        "Source Code": "https://github.com/MTG/essentia"
    },
    keywords='audio music sound dsp MIR',
    license='AGPLv3',
    platforms='any',
    classifiers=classifiers,
    setup_requires=build_requires,
    install_requires=build_requires,
    ext_modules=[module],
    cmdclass={
        'build_ext': EssentiaBuildExtension,
        'install_lib': EssentiaInstall
    }
)
