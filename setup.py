import shutil
import os
import glob
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib

library = None
PYTHON = sys.executable

def build_lib():
    os.system('rm -rf tmp')
    os.system('mkdir tmp')
    os.system('./packaging/build_3rdparty_static_debian.sh')
    os.system('%s waf configure --build-static --static-dependencies \
               --with-python --prefix=tmp' % PYTHON)
    os.system('%s waf' % PYTHON)
    os.system('%s waf install' % PYTHON)
    return glob.glob('tmp/lib/python*/*-packages/essentia')[0]


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
        library = build_lib();


VERSION = open('VERSION', 'r').read().strip('\n')

classifiers = [
    'License :: OSI Approved :: GNU Affero General Public License v3',
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research'
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

module = Extension('name', sources=[])

setup(
    name='essentia',
    version=VERSION,
    #version='2.1-beta5'
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
    install_requires=[
        'numpy',
        'six',
        'pyyaml'
    ],
    ext_modules=[module],
    cmdclass={
        'build_ext': EssentiaBuildExtension,
        'install_lib': EssentiaInstall
    }
)
