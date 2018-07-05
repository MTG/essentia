# Building Essentia's Python extension on Windows 

Python.org Windows binaries are compiled against an MS Visual C++ (MSVC) runtime, which version differs with different Python versions. Essentia's Python extension should be built with MSVC too because the MinGW C++ library builds won't be compatible with MSVC.

In this guide we use VirtualBox Windows machine.

## Preparing build environment
- Download Dev Virtual Machine: https://developer.microsoft.com/en-us/windows/downloads/virtual-machines
- Remove Visual Studio 2017 Community as we won't use it
- Install Build Tools for Visual Studio 2017: https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017
- Make sure VC++ v140 toolset is installed: Visual Studio Build Tools 2017 preferences --> check "VC++ 2015.3 v140 toolset for desktop (x86,x64)"
- Make sure Win10 SDK is also installed
- Install Python 3, add path to python.exe to %PATH% variable
- Patch ``distutils/msvc9compiler.py`` and ``distutils/msvccompiler.py`` (they are inside ``C:\Users\User\AppData\Local\Programs\Python\Python36-32\lib\distutils\`` folder): add a line ``import setuptools``
- Force upgrade of the setuptools package: python -m pip install -U pip setuptools
- Install pkg-config-lite (https://sourceforge.net/projects/pkgconfiglite/files/) to ``C:\workspace`` and add this path to %PATH% variable


- Download pre-built dependencies:
- Unpack into ```packaging\win32_3rdparty``` folder    


## Building Essentia
- Open "x86 Native Tools Command Prompt for VS 2017"
- Configure Essentia: ``python waf configure --with-python --msvc_targets="x86"``
- Compile and install: 
```
python waf
python waf install
```

The waf script will install: 
- library binary and headers to ``C:\Users\User\AppData\Local\Temp\include\essentia\`` and ``C:\Users\User\AppData\Local\Temp\lib\``
- Python extension to ``C:\Users\User\AppData\Local\Programs\Python\Python36-32\Lib\site-packages\essentia``

- In the case of import error in python due to missing DLLs, use Dependency Walker to find out missing dependencies.



## Building dependencies

We provide pre-build binaries for the required dependencies, however you can build them on your own. Essentia's waf build script expects dependencies to be inside the ```packaging\win32_3rdparty\builds``` folder with a proper subfolder name for each dependency. The prefixes inside the pkg-config files (*.pc) should be edited. See the files in the provided downloads as an example.

### FFMPEG

There are FFmpeg builds available online (http://ffmpeg.zeranoe.com/builds/), but unfortunately those do not include libavresample. Therefore, we have to build FFmpeg from scratch: 

- Install MSYS2: http://www.msys2.org/
- Install Yasm: download the yasm-*.exe file, rename to yasm.exe and copy to ``C:\workspace``

Run the script ``packaging/win32_3rdparty/prepare_ffmpeg_build.sh``. It takes care of everything. What it does is described below.

- Install pre-requisites in MSYS2 terminal: pacman -S tar make gcc diffutils

- Open "x86 Native Tools Command Prompt for VS 2017" and from this terminal open the Msys2 terminal (use the following flags to inherit %PATH% variable): ``C:\msys64\msys2_shell.cmd -msys -use-full-path``
- Go to the root Essentia folder (``cd /c/`` to access C:)
- ``cd packaging/win32_3rdparty/``
- Edit ``build_ffmpeg_msvc.sh`` to fix architecture if you are building for 64 bit (``--arch=x86_64``)
- Build ffmpeg: ``./build_ffmpeg_msvc.sh``

- Create *.lib export files

```
cd lib/
lib /def:avcodec-56.def /out:avcodec-56.lib
lib /def:avformat-56.def /out:avformat-56.lib
lib /def:avutil-54.def /out:avutil-54.lib
lib /def:avresample-2.def /out:avresample-2.lib
lib /def:swresample-1.def /out:swresample-1.lib
```

Rename: 
```
mv avcodec-56.lib avcodec.lib
mv avformat-56.lib avformat.lib
mv avutil-54.lib avutil.lib
mv avresample-2.lib avresample.lib
mv swresample-1.lib swresample.lib
```

Replace prefix in *.pc files to ```prefix=../packaging/win32_3rdparty/builds/ffmpeg-2.8.12```.

### FFTW

FFTW builds are available online, so we adapt those. Edit and run the ``prepare_fftw3_build.sh`` script; it takes care of everything. What it does is described below.

Download: 
- ftp://ftp.fftw.org/pub/fftw/fftw-3.3.3-dll32.zip (32-bit)
- ftp://ftp.fftw.org/pub/fftw/fftw-3.3.3-dll64.zip (64-bit)

- Generate .lib import file from .def file: http://www.fftw.org/install/windows.html

- Keep the dll filename as it is, but rename the rest of files from libfft3f-3.* to fft3f.* to make Essentia's waf script find them (waf does not add "lib" prefix to the filenames)


### Libsamplerate

For 32-bit, we also use prebuilt library files: https://github.com/MTG/essentia/tree/v2.1_beta4/packaging/win32_3rdparty/libsamplerate-0.1.8

- Rename files: libsamplerate-0.lib --> samplerate.lib


### Taglib

Build from source:
- https://oxygene.sk/2011/04/windows-binaries-for-taglib/
- https://github.com/taglib/taglib/blob/master/INSTALL.md

Steps:

- Install cmake: https://cmake.org/
- ``cmake  -DCMAKE_INSTALL_PREFIX=C:\users\User\Dev\lib``
- ``msbuild all_build.vcxproj /p:Configuration=Release``
- ``msbuild install.vcxproj``

When the build is done, the dll file is in ``taglib/Release`` folder.