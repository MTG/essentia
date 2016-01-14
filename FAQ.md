Frequently Asked Questions
==========================

How to compute music descriptors using Essentia?
------------------------------------------------

Because Essentia is a library you are very fexible in the ways you can compute descriptors out of audio:

- using [premade extractors out-of-box](doc/sphinxdoc/extractors_out_of_box.rst) (the easiest way without programming)
- using python (see [python tutorial](doc/sphinxdoc/python_tutorial.rst))
- writing your own C++ extractor (see the premade extractors as examples)


How to compile my own C++ code that uses Essentia?
--------------------------------------------------

Here is an example how to compile [standard_mfcc.cpp](https://github.com/MTG/essentia/blob/2.0.1/src/examples/standard_mfcc.cpp) example on Linux linking with a system-wide installation of Essentia (done by ```./waf install```) and all its dependencies. Modify to your needs. 

```
g++ -pipe -Wall -O2 -fPIC -I/usr/local/include/essentia/ -I/usr/local/include/essentia/scheduler/ -I/usr/local/include/essentia/streaming/  -I/usr/local/include/essentia/utils -I/usr/include/taglib -I/usr/local/include/gaia2 -I/usr/include/qt4 -I/usr/include/qt4/QtCore -D__STDC_CONSTANT_MACROS standard_mfcc.cpp -o standard_mfcc -L/usr/local/lib -lessentia -lfftw3 -lyaml -lavcodec -lavformat -lavutil -lsamplerate -ltag -lfftw3f -lQtCore -lgaia2
```

Alternatively, if you want to create and build your own examples, the easiest way is to add them to ```src/examples``` folder, modify ```src/examples/wscript``` file accordingly and use ```./waf configure --with-examples; ./waf``` to build them.

You can build your application using XCode (OSX) following [these steps](https://github.com/MTG/essentia/issues/58#issuecomment-38530548).


OSX static builds and templates (JUCE/VST and openFrameworks)
------------------------------------------------------------------------------------------------------------

Here you can find portable 32-bit static builds of the Essentia C++ library and its dependencies for OSX (thanks to Cárthach from GiantSteps) as well as templates for JUCE/VST and openFrameworks:

https://github.com/GiantSteps/Essentia-Libraries 



Linux/OSX static builds
-------------------

Follow the steps below to create static build of the library and executable example extractors.

Install additional tools required to build some of the dependencies. 

On Linux:
```
apt-get install yasm cmake
```

On OSX:
```
brew install yasm cmake
```

Prepare static builds for dependencies running a script (works both for Linux and OSX):
```
packaging/build_3rdparty_static_debian.sh
```

Alternatively, you can build each dependency apart running corresponding scripts inside ```packaging/debian_3rdparty``` folder:
```
cd packaging/debian_3rdparty
build_libav_nomuxers.sh
build_taglib.sh
build_fftw3.sh
build_libsamplerate.sh
build_yaml.sh
cd ../../
```

Build Essentia:
```
./waf configure  --with-static-examples
./waf
```

The static executables will be in the ```build/src/examples``` folder.


Cross-compiling for Windows on Linux
------------------------------------

Install Mingw-w64 GCC:
```
sudo apt-get install  gcc-mingw-w64 
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

Compiling Essentia to Javascript with Emscripten
------------------------------------------------
Use the instructions below to compile Essentia to Javascript. Among the dependencies, only FFTW3 is currently supported (see instructions to build it below). The rest of dependencies have not been tested, but they should work as well.

Install Emscripten following the [instructions](https://kripken.github.io/emscripten-site/docs/getting_started/downloads.html) on their website. If you downloaded the SDK manually, make sure to activate the Emscripten environment by executing `emsdk_env.sh`.
```
./path/to/emsdk_env.sh
```
Alternatively, you can install from Ubuntu/Debian repository (the environment will be activated by default).
```
sudo apt-get install emscripten
```

Get the latest FFTW3 source code, and prepare it for compilation and installation as an Emscripten system library and build it.
```
tar xf fftw-3.3.4.tar.gz
cd fftw-3.3.4
# Spawn a subshell to be able to use $EMSCRIPTEN in the command's args
emconfigure sh -c './configure --prefix=$EMSCRIPTEN/system/local/ CFLAGS="-Oz" --disable-fortran --enable-single'
emmake make
emmake make install
```

Finally, compile Essentia for Emscripten.
```
cd path/to/essentia
emconfigure sh -c './waf configure --prefix=$EMSCRIPTEN/system/local/ --lightweight=fftw --emscripten'
emmake ./waf
emmake ./waf install
```
Essentia is now built. If you want to build applications with Essentia and Emscripten, be sure to read their [tutorial](https://kripken.github.io/emscripten-site/docs/getting_started/Tutorial.html). Use the emcc compiler, preferably the ```-Oz``` option for size optimization, and include the static libraries for Essentia and FFTW as you would with source files. An example would be:
```
# Make sure your script can access the variable $EMSCRIPTEN
# (available to child processes of emconfigure and emmake)
LIB_DIR=$EMSCRIPTEN/system/local/lib
emcc -Oz -c application.cpp application.bc
emcc -Oz application.bc ${LIB_DIR}/libessentia.a ${LIB_DIR}/libfftw3f.a -o out.js
```


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
python test/src/unittest/all_tests.py -audioloader_streaming
```

Run a specific test
```
python test/src/unittest/all_tests.py audioloader_streaming
```


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

Training and running classifier models in Gaia
----------------------------------------------
In order to run classification in Essentia you need to prepare a classifier model in Gaia and run GaiaTransform algorithm configured to use this model. The example of using high-level models can be seen in the code of ```streaming_music_extractor```. Here we discuss the steps to be followed to train classifier models that can be used with this extractor.

1. Compute music descriptors using ```streaming_music_extractor``` for all audio files.
2. Install Gaia with python bindings.
3. Prepare json [groundtruth](https://github.com/MTG/gaia/blob/master/src/bindings/pygaia/scripts/classification/groundtruth_example.yaml) and [filelist](https://github.com/MTG/gaia/blob/master/src/bindings/pygaia/scripts/classification/filelist_example.yaml) files (see examples).
    - Groundtruth file maps identifiers for audio files (they can be paths to audio files or whatever id strings you want to use) to class labels. 
    - Filelist file maps these identifiers to the actual paths to the descriptor files for each audio track. 
4. Currently Gaia does not support loading descriptors in json format, as a workaround you can configure the extractor output to yaml format in Step 1, or run ```json_to_sig.py``` [conversion script](https://github.com/MTG/gaia/blob/master/src/bindings/pygaia/scripts/classification/json_to_sig.py).  
5. Run ```train_model.py``` script in Gaia ([here](https://github.com/MTG/gaia/blob/master/src/bindings/pygaia/scripts/classification/train_model.py)) with these groundtruth and filelist files. The script will create the classifier model file. 

6. The model file can now be used by a GaiaTransform algorithm inside ```streaming_music_extractor```. 

Note that using a specific classifier model implies that you are expected to give a pool with the same descriptor layout as the one used in training as an input to GaiaTransform Algorithm. 

The training script automatically creates an SVM model given a ground-truth dataset.  It allows to select for the best combination of SVM parameters (polynomial or RBF kernels, various gamma and C coefficients) in a grid search. In addition it also allows to do feature selection/preprocessing and select the best preprocessing among several that were identified as useful (e.g., all descriptors vs only spectral descriptors, or where to apply or not normalization; Currently, only means and variances are are used for descriptors summarized across frames). The combinations of parameters tested in a grid search are mentioned [in the code](https://github.com/MTG/gaia/blob/master/src/bindings/pygaia/scripts/classification/classification_project_template.yaml). Users are able to modify these parameters according to their needs by creating such a classification project file on their own.

To train the SVMs Gaia internally uses LibSVM library. For each combination of parameters in a grid search, 5-fold cross-validation evaluation is run splitting  ground-truth dataset into train and test splits and averaging results across folds (including the confusion matrix). After all combinations have been evaluated, the winner combination is selected according to the best accuracy and the final SVM classifier model is trained using *all* ground-truth data.

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

Using Essentia real-time
------------------------
You can use Essentia's streaming mode in real time feeding input audio frames to a network of algorithms via RingBufferInput. The output of the network can be consumed in real time using RingBufferOutput. 

As an example, see the code of [essentiaRT~](https://github.com/GiantSteps/MC-Sonaar/tree/master/essentiaRT~). 

- [EssentiaOnset.cpp#L63](https://github.com/GiantSteps/MC-Sonaar/blob/master/essentiaRT~/EssentiaOnset.cpp#L63)
- [EssentiaOnset.cpp#L112](https://github.com/GiantSteps/MC-Sonaar/blob/master/essentiaRT~/EssentiaOnset.cpp#L112)
- [main.cpp#L74](https://github.com/GiantSteps/MC-Sonaar/blob/master/essentiaRT~/main.cpp#L74)

You can also use Essentia's standard mode for real-time computations. 

Not all algorithms available in the library are suited for real-time analysis due to their computational complexity. Some complex algorithms, such as BeatTrackerDegara, BeatTrackerMultiFeatures, and PredominantMelody, require large segments of audio in order to function properly.

Make sure that you do not reconfigure an algorithm (from the main UI thread, most likely) while an audio callback (from an audio thread) is currently being called, as the algorithms are not thread-safe.


libessentia.so is not found after installing from source
--------------------------------------------------------
The library is installed into /usr/local and your system does not search for shared libraries there. [Configure your paths properly](http://unix.stackexchange.com/questions/67781/use-shared-libraries-in-usr-local-lib).


Building standalone Essentia Vamp plugin
----------------------------------------

It is possible to create a standalone binary for Essentia's Vamp plugin (works for Linux and OSX).

```
./waf configure --build-static --with-vamp --mode=release --lightweight= --fft=KISS
./waf
```

The resulting binary (```build/src/examples/libvamp_essentia.so``` on Linux, ```build/src/examples/libvamp_essentia.dylib``` on OSX) is a lightweight shared library that can be distributed as a single file without requirement to install Essentia's dependencies on the target machine.







