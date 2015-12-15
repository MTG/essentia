.. How-to install Essentia

Installing Essentia
===================

Mac OSX
-------
The easiest way to install Essentia on OSX is by using `our Homebrew formula <https://github.com/MTG/homebrew-essentia>`_. You will need to install `Homebrew package manager <http://brew.sh>`_ first (and there are other good reasons to do it apart from Essentia).


Linux
-----
We are currently preparing deb packages for Ubuntu and Debian. Meanwhile, you need to compile Essentia from source (see below).


Windows, Android
----------------
Cross-compile Essentia from Linux/OSX (see below).


.. Installing Essentia is easily done using the precompiled packages that you can find on the
.. `MIR-dev Essentia download page <http://static.mtg.upf.edu/mir-dev-download/essentia/>`_.
.. Packages are available for Debian/Ubuntu, Windows and Mac OS X.

.. These packages contain development headers to integrate Essentia in a C++ application, Python
.. bindings to be able to work in a Matlab-like environment, and some C++ examples and extractors.

.. Those who wish to write new descriptors can do it using the provided development headers,
.. but it is highly recommended though that they compile Essentia from source.


Compiling Essentia from source
==============================

Essentia depends on (at least) the following libraries:
 - `FFTW <http://www.fftw.org>`_: for the FFT implementation *(optional)*
 - `libavcodec/libavformat/libavutil/libavresample <http://ffmpeg.org/>`_ (from the FFmpeg/LibAv project): for loading/saving any type of audio files *(optional)*
 - `libsamplerate <http://www.mega-nerd.com/SRC/>`_: for resampling audio *(optional)*
 - `TagLib <http://developer.kde.org/~wheeler/taglib.html>`_: for reading audio metadata tags *(optional)*
 - `LibYAML <http://pyyaml.org/wiki/LibYAML>`_: for YAML files input/output *(optional)*
 - `Gaia <https://github.com/MTG/gaia>`_: for using SVM classifier models *(optional)*


Installing dependencies on Linux
--------------------------------

You can install those dependencies on a Debian/Ubuntu system from official repositories using the commands provided below::

  sudo apt-get install build-essential libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev libtag1-dev

In order to use python bindings for the library, you might also need to install python-numpy-dev (or python-numpy on Ubuntu) and python-yaml for YAML support in python::

  sudo apt-get install python-numpy-dev python-numpy python-yaml


Note that, depending on the version of Essentia, different versions of libav* and libtag1-dev packages are required. See `release notes for official releases <https://github.com/MTG/essentia/releases>`_. In the case of Essentia's **master branch**, the required version of TagLib (libtag1-dev) is greater or equal to ``1.9``. The required version of LibAv (``libavcodec-dev``, ``libavformat-dev``, ``libavutil-dev`` and ``libavresample-dev``) is greater or equal to ``10``. The appropriate versions are distributed in Ubuntu Utopic (14.10) repository, and in Debian wheezy-backports.


Installing dependencies on Mac OS X
-----------------------------------

Install Command Line Tools for Xcode. Even if you install Xcode from the app store you must configure command-line compilation by running::

  xcode-select --install

Install `Homebrew package manager <http://brew.sh>`_.

Insert the Homebrew directory at the top of your PATH environment variable by adding the following line at the bottom of your ``~/.profile`` file::

  export PATH=/usr/local/bin:/usr/local/sbin:$PATH

Install prerequisites::

  brew install pkg-config gcc readline sqlite gdbm freetype libpng

Install Essentia's dependencies::

  brew install libyaml fftw ffmpeg libsamplerate libtag

`Install python environment using Homebrew <http://docs.python-guide.org/en/latest/starting/install/osx>`_ (Note that you are advised to do as described here and there are `good reasons to do so <http://docs.python-guide.org/en/latest/starting/install/osx/>`_. You will most probably encounter installation errors when using python/numpy preinstalled with OSX.)::

  brew install python --framework
  pip install ipython numpy matplotlib pyyaml



Compiling Essentia
------------------

Once your dependencies are installed, you can proceed to compiling Essentia. Download Essentia's source code at `Github <https://github.com/MTG/essentia>`_.  Due to different dependencies requirement (see `release notes for official releases <https://github.com/MTG/essentia/releases>`_), make sure to download the version compatible with your system:
 - **2.1 beta2** is the official version currently recommended to install, it is supported on **Ubuntu 14.04 LTS or higher**, **Debian Jessie or higher** and **OSX**.
 - **master** branch is the most updated version of Essentia in development, it is supported on **Ubuntu 14.10 or higher**, **Debian Jessie or higher** and **OSX**.


Go into its source code directory and start by configuring it::

  ./waf configure --mode=release --build-static --with-python --with-cpptests --with-examples --with-vamp --with-gaia

Use the keys:
 - ``--with-python`` to enable python bindings,
 - ``--with-examples`` to build examples based on the library,
 - ``--with-vamp`` to build vamp plugin wrapper,
 - ``--with-gaia`` to build with Gaia library support.

NOTE: you must *always* configure at least once before building!

The following will give you a full list of options::

  ./waf --help

To compile everything you've configured::

  ./waf

To install the C++ library and the python bindings (if configured successfully; you might need to run this command with sudo)::

  ./waf install

All built examples (including the out-of-box features extractors) will be located in ``build/src/examples/`` folder, as well as the vamp plugin file ``libvamp_essentia.so``. In order to use the plugin you will need to place this file to the the standard vamp plugin folder of your system (such as ``/usr/local/lib/vamp/`` on Linux).


Running tests (optional)
------------------------
If you want to assure that Essentia works correctly, do the tests.

To run the C++ base unit tests (only test basic library behavior)::

  ./waf run_tests

To run the python unit tests (include all unittests on algorithms, need python bindings installed first)::

  ./waf run_python_tests


Building documentation (optional)
---------------------------------

All documentation is provided on the official website of Essentia library. To generate it by your own follow the steps below.

Install doxigen and pip, if you are on Linux::

  sudo apt-get install doxygen python-pip

Install additiona dependencies (you might need to run this command with sudo)::

  sudo pip install sphinx pyparsing sphinxcontrib-doxylink docutils

Make sure to install Essentia with python bindings and run::

  ./waf doc

Documentation will be located in ``doc/sphinxdoc/_build/html/`` folder.


Building Essentia on Windows
----------------------------

Essentia C++ library and extractors based on it can be compiled and run correctly on Windows, but python bindings are currently not supported. The easiest way to build Essentia is by `cross-compilation on Linux using MinGW <https://github.com/MTG/essentia/blob/master/FAQ.md#cross-compiling-for-windows-on-linux>`_. However the resulting library binaries are only compatible within C++ projects using MinGW compilers, and therefore they are not compatible with Visual Studio. If you want to use Visual Studio, there is no project readily available, so you will have to setup one yourself and compile the dependencies too.


Building Essentia on Android
----------------------------

A lightweight version of Essentia can be `cross-compiled for Android <https://github.com/MTG/essentia/blob/master/FAQ.md#cross-compiling-for-android>`_ from Linux or Mac OSX.


Using pre-trained high-level models in Essentia
-----------------------------------------------

Essentia includes a number of `pre-trained classifier models for genres, moods and instrumentation
<algorithms_overview.html#other-high-level-descriptors>`_. In order to use them you need to:

* Install `Gaia2 library <https://github.com/MTG/gaia/blob/master/README.md>`_ (supported on Linux/OSX)
* Build Essentia with examples and Gaia (--with-examples --with-gaia)
* Use ``streaming_extractor_music`` (see `detailed documentation <streaming_extractor_music.html>`_)

You can `train your own classifier models <https://github.com/MTG/essentia/blob/master/FAQ.md#training-and-running-classifier-models-in-gaia>`_.

