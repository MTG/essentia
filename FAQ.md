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

Alternatively, you can build each dependency apart running corresponding scripts inside ```packaging/debian_3rdparty``` folder:
```
cd packaging/debian_3rdparty
build_libav_nomuxers.sh
build_taglib.sh
build_fftw3.sh
build_libsamplerate.sh
build_yaml.sh
...
cd ../../
```

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

Alternatively to steps 3-5, you can use a simplified [script](https://github.com/MTG/gaia/blob/master/src/bindings/pygaia/scripts/classification/train_model_from_sigs.py) that trains a model given a folder with sub-folders corresponding to class names and containing descriptor files for these classes. 

Note that using a specific classifier model implies that you are expected to give a pool with the same descriptor layout as the one used in training as an input to GaiaTransform Algorithm. 

### How it works
To train the SVMs Gaia internally uses [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) library. The training script automatically creates an SVM model given a ground-truth dataset using the best combination of parameters for data preprocessing and SVM that it can find in a grid search. Testing all possible combinations the script conducts a 5-fold cross-validation for each one of them: The ground-truth dataset is randomly split into train and test sets, the model is trained on the train set and is evaluated on the test set. Results are averaged across 5 folds including the confusion matrix. After all combinations of parameters have been evaluated, the winner combination is selected according to the best accuracy obtained in cross-validation and the final SVM classifier model is trained using *all* ground-truth data. See the "Cross-validation and Grid-search" section in the [practical guide to SVM classification](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) for more details.

The combinations of parameters tested in a grid search by default are mentioned [in the code](https://github.com/MTG/gaia/blob/master/src/bindings/pygaia/scripts/classification/classification_project_template.yaml). Users are able to modify these parameters according to their needs by creating such a classification project file on their own.

The parameters include:
- SVM kernel type: polynomial or RBF
- SVM type: currently only C-SVC
- SVM C and gamma parameters
- preprocessing type:
    - use all descriptors, no preprocessing
    - use ```lowlevel.*``` descriptors only
    - discard energy bands descriptors (```*barkbands*```, ```*energyband*```, ```*melbands*```, ```*erbbands*```)
    - use all descriptors, normalize values
    - use all descriptors, normalize and gaussianize values
- number of folds in cross-validation: 5 by default

In the preprocessing stage, training script loads all descriptor files according to the preprocessing type. Additionally, a number of descriptors are always ignored, including all ```metadata*``` that is the information not directly associated with audio analysis. The ```*.dmean```, ```*.dvar```, ```*.min```, ```*.max```, ```*.cov``` descriptors are also ignored, and therefore, currently only means and variances are used for descriptors summarized across frames. Non-numerical descriptors are enumerated (```tonal.chords_key```, ```tonal.chords_scale```, ```tonal.key_key```, ```tonal.key_scale```).

Note that cross-validation script splits the ground-truth dataset into train and test sets randomly. In the case of music classification tasks one may want to assure artist/album filtering (that is, no artist/album occures in the test set if it occures in train set). Current way to achieve it is to ensure that the whole input dataset contains only one item per artist/album. Alternatively, you can adapt the scripts to suit your needs.

### How to train an SVM model with a different set of parameters 
Our training script generates a single model retrained on the whole dataset with the best parameters combination from the grid search. However, you may want to generate new models with custom parametrizations. Imagine, for instance, that you need a model that runs on a lighter set of features despite the accuracy drop, or that you believe that a different parameters set can improve results for your particular scenario.

In order to generate a model given the `<project_file>` and your chosen `<param_file>` from the results folder, execute the following python lines,

```
from gaia2.scripts.classification.retrain_model import retrainModel
retrainModel(project_file, param_file, output_file)

```
This creates a Gaia model and saves it into `<output_file>`. 

*Also, note that the `retrain_model` can be called as a command line program.*

### How to choose a parameter configuration
At the end of the training process, a file called `<project_name>.report.csv` is created. It provides a ranking in terms of accuracy and normalized accuracy as well as the standard deviation between folds for every set of parameters. By having a look at this file you can get some insights about which parameters to try. You can, for instance, estimate the expected accuracy drop if you decide to go for a configuration with a smaller set of descriptors.


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



