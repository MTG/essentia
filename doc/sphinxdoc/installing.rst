.. How-to install Essentia

.. Installing Essentia
.. ===================
..
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

You can install those dependencies on a Debian/Ubuntu system from official repositories using the commands provided below. Note that, depending on the version of Essentia, different versions of libav* and libtag1-dev packages are required. See Github release notes on the download page.

In the case of Essentia 2.1, the required version of TagLib (libtag1-dev) is greater or equal to ``1.9``. The suitable version is distributed with Ubuntu Trusty (14.04 LTS). If you are using the latest stable Debian (Wheezy), you might want to install it from `wheezy-backports <https://wiki.debian.org/Backports>`_ repository. The required version of LibAv (``libavcodec-dev``, ``libavformat-dev``, ``libavutil-dev`` and ``libavresample-dev``) is greater or equal to ``10``. The appropriate versions are distributed in Ubuntu Utopic (14.10) repository, and in Debian wheezy-backports.

**Essentia 2.1 on Ubuntu 14.10**::

  sudo apt-get install build-essential libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev libtag1-dev

In order to use python bindings for the library, you might also need to install python-numpy-dev or python-numpy on Ubuntu::

  sudo apt-get install python-numpy-dev python-numpy

Install support for YAML files input/output in python (optional, make sure to have libyaml installed first)::

  sudo apt-get install python-pip
  pip install pyyaml



Installing dependencies on Mac OS X
-----------------------------------

Install a scientific python environment first:

1. install Command Line Tools for Xcode. Even if you install Xcode from the app store you must configure command-line compilation by running: ``xcode-select --install``
2. install homebrew (package manager): http://brew.sh
3. install prerequisites: ``brew install pkg-config gcc readline sqlite gdbm freetype libpng``
4. install python: ``brew install python --framework``
5. install ipython and numpy, matplotlib, and pyyaml: ``pip install ipython numpy matplotlib pyyaml``
6. when launching ipython, use:

  a. ``ipython --pylab``    if you have matplotlib   >= 1.3
  b. ``ipython --pylab=tk`` if you have matplotlib < 1.3

Note that you are advised to install python environment **as described here**, i.e., via homebrew and pip. You will most probably encounter installation errors when using
python/numpy preinstalled with OSX 10.9. More details can be found at https://github.com/mxcl/homebrew/wiki/Homebrew-and-Python

Then run::

  brew install libyaml fftw ffmpeg libsamplerate libtag



Compiling Essentia
------------------

Once your dependencies are installed, you can compile Essentia (the library) by going into its
directory and start by configuring it::

  ./waf configure --mode=release --with-python --with-cpptests --with-examples --with-vamp --with-gaia

Use the keys:
   ``--with-python`` to enable python bindings,
   ``--with-examples`` to build examples based on the library,
   ``--with-vamp`` to build vamp plugin wrapper,
   ``--with-gaia`` to build with Gaia library support.

NOTE: you must *always* configure at least once before building!

The following will give you a list of options::

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

Essentia does compile and run correctly on Windows (python bindings were not tested). The easiest way to build Essentia is by cross-compilation on Linux using MinGW: https://github.com/MTG/essentia/blob/master/FAQ.md#cross-compiling-for-windows-on-linux

However, if you want to use Visual Studio, there is no project readily available, so you will have to setup one yourself and compile the dependencies too. It appears that binaries for the library generated by cross-compilation are not compatible with Visual Studio.



Using pre-trained high-level models in Essentia
-----------------------------------------------

Essentia includes a number of `pre-trained classifier models for genres, moods and instrumentation
<algorithms_overview.html#other-high-level-descriptors>`_. In order to use them you need to:

* Install Gaia2 library (supported on Linux/OSX): https://github.com/MTG/gaia/blob/master/README.md
* Build Essentia with examples and Gaia (--with-examples --with-gaia)
* Use ``streaming_extractor_music`` (see `detailed documentation <streaming_extractor_music.html>`_)

You can also use classifier models trained by your own: https://github.com/MTG/essentia/blob/master/FAQ.md#training-and-running-classifier-models-in-gaia

