# Frequently Asked Questions

libessentia.so is not found after installing from source
--------------------------------------------------------
The library is installed into `/usr/local` and your system does not search for shared libraries there. [Configure your paths properly](http://unix.stackexchange.com/questions/67781/use-shared-libraries-in-usr-local-lib).


Build Essentia on Ubuntu 14.04 or earlier
-----------------------------------------
As it is noted in the [installation guide](http://essentia.upf.edu/documentation/installing.html), Essentia is only compatible with LibAv versions greater or equal to 10. The appropriate versions are distributed since Ubuntu 14.10 and Debian Jessie. If you have an earlier system (e.g., Ubuntu 14.04), you can choose one of the two options:

- upgrade your system which is recommended to do anyways in the long-term (e.g., to the latest Ubuntu LTS 16.04)
- install the LibAv dependency from source

To install LibAv from source:

- If you have installed LibAv before, remove it so that it does not mess up Essentia installation 
    ```
    sudo apt-get remove libavcodec-dev libavformat-dev libavutil-dev libavresample-dev
    ```
- Download and unpack [LibAv source code](https://libav.org/download/)
- Configure and build LibAv. The library will be installed to ```/usr/local```.
    ```
    ./configure --disable-yasm --enable-shared
    make
    sudo make install
    ```
- [Configure and build Essentia](http://essentia.upf.edu/documentation/installing.html#compiling-essentia)


Linux/OSX static builds
-----------------------

Follow the steps below to create static build of the library and executable example extractors.
-
Install additional tools required to build some of the dependencies. 

On Linux:
```
apt-get install yasm cmake
```

On OSX:
```
brew install yasm cmake wget
```

Prepare static builds for dependencies running a script (works both for Linux and OSX):
```
packaging/build_3rdparty_static_debian.sh
```

Use ```--with-gaia``` flag to include Gaia.

Use ```--with-tensorflow``` flag to include TensorFlow.

Alternatively, you can build each dependency apart running the corresponding scripts inside ```packaging/debian_3rdparty``` folder:
```
cd packaging/debian_3rdparty
build_<dependency>.sh
...
cd ../../
```
Note that you can selectively build dependencies depending on the required Essentia algorithms.

Build Essentia:
```
./waf configure  --with-static-examples
./waf
```

The static executables will be in the ```build/src/examples``` folder.


Building lightweight Essentia with reduced dependencies 
-----------------------------------------------------
Since version 2.1, build scripts can be configured to ignore 3rdparty dependencies required by Essentia in order to create a striped-down version of the library.  Use  ```./waf configure``` command with the ```--lightweight``` flag to provide the list of 3rdparty dependencies to be included. For example, the command below will configure to build Essentia avoiding all dependencies except fftw:
```
./waf configure --lightweight=fftw
```

Avoid all dependencies including fftw and build with KissFFT instead (BSD, included in Essentia therefore no external linking needed, cross-platform):

```
./waf configure --lightweight= --fft=KISS
```

Avoid all dependencies and build with Accelerate FFT (native on OSX/iOS):

```
./waf configure --lightweight= --fft=ACCELERATE
```

It is also possible to specify algorithms to be ignored using the ```--ignore-algos``` flag, although you need to take care that the ignored algorithm are not required by any of the algorithms and examples that will be compiled. 

Note, that Essentia includes in its code the Spline library (LGPLv3) which is used by Spline and CubicSpline algorithms and is built by default. To ignore this library, use the following flag in ```./waf configure``` command:
```
--ignore-algos=Spline,CubicSpline
```

For more details on the build flags, run:
```
./waf --help
```


Cross-compiling for Windows on Linux
------------------------------------

Install Mingw-w64 GCC:
```
sudo apt-get install g++-mingw-w64
```

Build all dependencies (similarly to Linux static builds, make sure you have required tools installed):
```
./packaging/build_3rdparty_static_win32.sh
```

Build Essentia with static examples:
```
./waf configure --with-static-examples --cross-compile-mingw32
./waf
```


Cross-compiling for Android
---------------------------

A lightweight version of Essentia can be compiled using the ```--cross-compile-android``` flag. It requires reducing the dependencies to a bare minimum using KissFFT library for FFT. Specify the installation prefix with ```--prefix``` flag. Update the ```PATH``` variable to point to where you have your Android Standalone Toolchain.

```
export PATH=~/Dev/android/toolchain/bin:$PATH;
./waf configure --cross-compile-android --lightweight= --fft=KISS --prefix=/Users/carthach/Dev/android/modules/essentia
./waf
./waf install
```


Cross-compiling for iOS
-----------------------
A lightweight version of Essentia for iOS can be compiled using the ```--cross-compile-ios``` flag. It requires reducing the dependencies to a bare minimum using Accelerate Framework for FFT. 

```
./waf configure --cross-compile-ios --lightweight= --fft=ACCELERATE --build-static
```

You can also compile it for iOS simulator (so that you can test on your desktop) using ```--cross-compile-ios-sim``` flag.


Compiling Essentia to ASM.js or WebAssembly using Emscripten
------------------------------------------------------------
Use the instructions below to compile Essentia to intermediate [LLVM](https://llvm.org/) or [ASM.js](http://asmjs.org/) and [WebAssembly](https://webassembly.org/)(WASM) targets using [Emscripten](https://emscripten.org/). You can build Essentia with or without third party dependencies. Among the dependencies, only FFTW3 is currently supported (see instructions to build it below). The rest of dependencies have not been tested, but they should work as well. A lightweight WASM build of Essentia is used in our dedicated JavaScript wrapper [Essentia.js](https://essentia.upf.edu/essentiajs) which uses KISS FFT instead of FFTW3.

- Install the latest stable Emscripten release following the [instructions](https://emscripten.org/docs/getting_started/downloads.html) on their website. If you downloaded the SDK manually, make sure to activate the Emscripten environment by executing `emsdk_env.sh`.


(Optional with third party dependecies)
- Get the latest [FFTW3](http://www.fftw.org/) source code, and prepare it for compilation and installation as an Emscripten system library and build it.
  
```bash
tar xf fftw-3.3.4.tar.gz
cd fftw-3.3.4
# Spawn a subshell to be able to use $EMSCRIPTEN in the command's args
emconfigure sh -c './configure --prefix=$EMSCRIPTEN/system/local/ CFLAGS="-Oz" --disable-fortran --enable-single'
emmake make
emmake make install
```

- Finally, compile Essentia with Emscripten as an LLVM target which can be further used for linking with your application code.
  
```bash
cd path/to/essentia
# for using KISS FFT
emconfigure sh -c './waf configure --prefix=$EMSCRIPTEN/system/local/ --lightweight=KISS --emscripten'
# OR
# for using FFTW
emconfigure sh -c './waf configure --prefix=$EMSCRIPTEN/system/local/ --lightweight=FFTW --emscripten'
emmake ./waf
emmake ./waf install
```
Essentia is now built. If you want to build applications with Essentia and Emscripten, be sure to read their [tutorial](https://kripken.github.io/emscripten-site/docs/getting_started/Tutorial.html). Essentia.js Github [repository](https://github.com/MTG/essentia.js) also has some nice set of examples for you to get started. Use the emcc compiler, preferably the ```-Oz``` option for size optimization, and include the static libraries for Essentia and FFTW as you would with source files. An example would be:

```bash
# Make sure your script can access the variable $EMSCRIPTEN
# (available to child processes of emconfigure and emmake)
LIB_DIR=$EMSCRIPTEN/system/local/lib
emcc -Oz -c application.cpp application.bc
emcc -Oz application.bc ${LIB_DIR}/libessentia.a ${LIB_DIR}/libfftw3f.a -s WASM=1 -o out.js
```
Alternatively you could also build your applicaitons for asm.js targets by changing the flag `-s WASM=0`.

You can also find some examples of interfacing your Essentia cpp code to JavaScript [here](https://github.com/MTG/essentia.js/blob/master/docs/tutorials/2.%20Building%20from%20Source.md#writing-custom-essentia-c-extractor-and-cross-compile-to-js).

OSX static builds and templates (JUCE/VST and openFrameworks)
-------------------------------------------------------------

Here you can find portable 32-bit static builds of the Essentia C++ library and its dependencies for OSX (thanks to Cárthach from GiantSteps) as well as templates for JUCE/VST and openFrameworks:

https://github.com/GiantSteps/Essentia-Libraries


Building standalone Essentia Vamp plugin
----------------------------------------

It is possible to create a standalone binary for Essentia's Vamp plugin (works for Linux and OSX).

```
./waf configure --build-static --with-vamp --mode=release --lightweight= --fft=KISS
./waf
```

The resulting binary (```build/src/examples/libvamp_essentia.so``` on Linux, ```build/src/examples/libvamp_essentia.dylib``` on OSX) is a lightweight shared library that can be distributed as a single file without requirement to install Essentia's dependencies on the target machine.


Running tests
-------------
In the case you want to assure correct working of Essentia, do the tests.

The most important test is the basetest, it should never fail: 
```
./build/basetest
```

Run all python tests: 
```
./waf run_python_tests
```
    
Run all tests except specific ones:
```
python test/src/unittests/all_tests.py -audioloader_streaming
```

Run a specific test
```
python test/src/unittests/all_tests.py audioloader_streaming
```


Writing tests
-------------
It is manadatory to write python unit tests when developing new algorithms to be included in Essentia. The easiest way to start writing a test is to adapt [existing examples](https://github.com/MTG/essentia/tree/master/test/src/unittests).

All unit tests for algorithms are located in ```test/src/unittests``` folder. They are organized by sub-folders similarly to the code for the algorithms. 

Typically tests include:

- Tests for invalid parameters
- Tests for incorrect inputs
- Tests for empty, silence or constant-value inputs
- Tests for simulated data inputs for which the output is known
- Regression tests for real data inputs for which the reference output was previously computed.
    - These tests are able to detect if there was a change in output values according to the expected reference. The reference is not necessarily a 100% correct ground truth. In many case the reference is built using an earlier version of the same algorithm being tested or is obtained from other software.

A number of assert methods are available: 

- ```assertConfigureFails``` (test if algorithm configuration fails)
- ```assertComputeFails``` (test if algorithm's compute method fails)
- ```assertRaises``` (test if exception is raised)
- ```assertValidNumber``` (test if a number is not NaN nor Inf)
- ```assertEqual```, ```assertEqualVector```, ```assertEqualMatrix``` (test if observed and expected values are equal)
- ```assertAlmostEqualFixedPrecision```, ```assertAlmostEqualVectorFixedPrecision``` (test if observed and expected values are approximately equal by computing the difference, rounding to the given number on decimal places, and comparing to zero)
- ```assertAlmostEqual```, ```assertAlmostEqualVector```, ```assertAlmostEqualMatrix``` (test if observed and expected values are approximately equal according to the given allowed relative error.
- ```assertAlmostEqualAbs```, ```assertAlmostEqualVectorAbs``` (test if the difference between observed and expected value is lower than then the given absolute threshold)


How to compile my own C++ code that uses Essentia?
--------------------------------------------------

Here is an example how to compile [standard_mfcc.cpp](https://github.com/MTG/essentia/blob/2.0.1/src/examples/standard_mfcc.cpp) example on Linux linking with a system-wide installation of Essentia (done by ```./waf install```) and all its dependencies. Modify to your needs. 

```
g++ -pipe -Wall -O2 -fPIC -I/usr/local/include/essentia/ -I/usr/local/include/essentia/scheduler/ -I/usr/local/include/essentia/streaming/  -I/usr/local/include/essentia/utils -I/usr/include/taglib -I/usr/local/include/gaia2 -I/usr/include/qt4 -I/usr/include/qt4/QtCore -D__STDC_CONSTANT_MACROS standard_mfcc.cpp -o standard_mfcc -L/usr/local/lib -lessentia -lfftw3 -lyaml -lavcodec -lavformat -lavutil -lsamplerate -ltag -lfftw3f -lQtCore -lgaia2
```

Alternatively, if you want to create and build your own examples, the easiest way is to add them to ```src/examples``` folder, modify ```src/examples/wscript``` file accordingly and use ```./waf configure --with-examples; ./waf``` to build them.

If you would also like to use [waf](https://waf.io/) in your application as we do, we provide an [example waf template using Essentia](https://github.com/MTG/essentia-project-template/).

You can build your application using XCode (OSX) following [these steps](https://github.com/MTG/essentia/issues/58#issuecomment-38530548).


How to compute music descriptors using Essentia?
------------------------------------------------

Because Essentia is a library you are very fexible in the ways you can compute descriptors out of audio:

- using [premade extractors out-of-box](extractors_out_of_box.html) (the easiest way without programming)
- using python (see [python tutorial](python_tutorial.html))
- writing your own C++ extractor (see the premade extractors as examples)


How to know which other Algorithms an Algorithm uses?
-----------------------------------------------------

The most obvious answer is: by reading its code. However, it is also possible to generate such a list automatically. 

Running the python script ```src/examples/python/show_algo_dependencies.py``` will output a list of all intermediate Algorithms created within each Algorithm in Essentia. It utilizes the logging framework and watches for messages generated by AlgorithmFactory at the moment of running ```create()``` method for each internal algorithm.  

Note, that you cannot be sure this list of dependencies is 100% correct as the script simply instantiates each algorithm to test for its dependencies, but does not run the ```compute``` stage. It is up to developers conscience to keep instantiations in a correct place, and if an Algorithm is being created on the ```compute``` stage, it will be unnoticed.

## How many algorithms are in Essentia?

The amount of algorithms counting streaming and standard mode separately:
```
python src/examples/python/show_algo_dependencies.py > /tmp/all.txt
cat /tmp/all.txt | grep -- "---------- " | wc -l
```

The amount of algorithms counting both modes as one algorithm:
```
python src/examples/python/show_algo_dependencies.py > /tmp/all.txt
cat /tmp/all.txt | grep -- "---------- " | cut -c 12- | sed s/"streaming : "// | sed s/"standard : "// | sed s/" ----------"// | sort -u | wc -l
```


Using Essentia real-time
------------------------
You can use Essentia's streaming mode in real time feeding input audio frames to a network of algorithms via RingBufferInput. The output of the network can be consumed in real time using RingBufferOutput. 

As an example, see the code of [essentiaRT~](https://github.com/GiantSteps/MC-Sonaar/tree/master/essentiaRT~). 

- [EssentiaOnset.cpp#L70](https://github.com/GiantSteps/MC-Sonaar/blob/master/essentiaRT~/EssentiaOnset.cpp#L70)
- [EssentiaOnset.cpp#L127](https://github.com/GiantSteps/MC-Sonaar/blob/master/essentiaRT~/EssentiaOnset.cpp#L127)
- [main.cpp](https://github.com/GiantSteps/MC-Sonaar/blob/master/essentiaRT~/main.cpp)

You can also use Essentia's standard mode for real-time computations. 

Not all algorithms available in the library are suited for real-time analysis due to their computational complexity. Some complex algorithms, such as BeatTrackerDegara, BeatTrackerMultiFeatures, and PredominantMelody, require large segments of audio in order to function properly.

Make sure that you do not reconfigure an algorithm (from the main UI thread, most likely) while an audio callback (from an audio thread) is currently being called, as the algorithms are not thread-safe.


Essentia Music Extractor
------------------------

### Converting descriptor files to CSV

Many researchers are still unfamiliar with [JSON](https://en.wikipedia.org/wiki/JSON) and instead commonly use [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file format. We have provided a python script that can convert a bunch of input JSON descriptor files (produced by Music Extractor or Freesound extractor) into a CSV file, where each raw represents analysis results for a particular audio recording. 

For more help, run: 
```
python src/examples/python/json_to_csv.py -h
```

Example command that merges analysis for two recordings, ignoring a bunch of descriptors:
```
python src/examples/python/json_to_csv.py -i /tmp/1.json /tmp/2.json -o /tmp/foo.csv --include metadata.audio_properties.* metadata.tags.musicbrainz_recordingid.0 lowlevel.* rhythm.* tonal.* --ignore *.min *.min.* *.max *.max.* *.dvar *.dvar2 *.dvar.* *.dvar2.* *.dmean *.dmean2 *.dmean.* *.dmean2.* *.cov.* *.icov.* rhythm.beats_position.*  --add-filename
```



