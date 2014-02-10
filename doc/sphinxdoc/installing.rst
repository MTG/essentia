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
 - `FFTW <http://www.fftw.org>`_: for the FFT implementation
 - `libavcodec <http://ffmpeg.org/>`_ (from the FFmpeg project): for loading/saving any type of audio files *(optional)*
 - `libsamplerate <http://www.mega-nerd.com/SRC/>`_: for resampling audio *(optional)*
 - `TagLib <http://developer.kde.org/~wheeler/taglib.html>`_: for reading audio metadata tags *(optional)*
 - `LibYAML <http://pyyaml.org/wiki/LibYAML>`_: for YAML files input/output *(optional)*


Installing dependencies on Linux
--------------------------------

You can install those dependencies on a Debian/Ubuntu system using the following command::

  sudo apt-get install build-essential libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev python-dev libsamplerate0-dev libtag1-dev

In order to use python bindings for the library, you might also need to install python-numpy-dev or python-numpy on Ubuntu::

  sudo apt-get install python-numpy-dev python-numpy


Installing dependencies on Mac OS X
-----------------------------------

Install a scientific python environment first:

1. install Command Line Tools for Xcode: https://github.com/mxcl/homebrew/wiki/Installation
2. install homebrew (package manager): http://brew.sh/
3. install prerequisites: ``brew install pkg-config gfortran readline sqlite gdbm freetype libpng``
4. install python: ``brew install python --framework``
5. install ipython and numpy: ``pip install ipython numpy``
6. install matplotlib: ``pip install matplotlib``
7. when launching ipython, use:

  a. ``ipython --pylab``    if you have matplotlib >= 1.3
  b. ``ipython --pylab=tk`` if you have matplotlib < 1.3

Note that you are advised to install python environment **as described here**, i.e., via homebrew and pip. You will most probably encounter installation errors when using
python/numpy preinstalled with OSX 10.9.

More details can be found at https://github.com/mxcl/homebrew/wiki/Homebrew-and-Python

Then run::

  brew install libyaml fftw ffmpeg libsamplerate libtag


Installing dependencies on Windows
----------------------------------

Essentia does compile and run correctly on Windows, however there is no Visual
Studio project readily available, so you will have to setup one yourself and
compile the dependencies too. We will be working on Windows installer in the near future. 


Additional dependencies (python, all platforms)
-----------------------------------------------

To build the documentation you will also need the following dependencies::

  pip install sphinx pyparsing==1.5.7 sphinxcontrib-doxylink docutils

Other useful dependencies::

  pip install pyyaml   # make sure to have libyaml installed first

You might need to install pip before, if you are on Linux::
  
  sudo apt-get install python-pip



Compiling Essentia
------------------

Once your dependencies are installed, you can compile Essentia (the library) by going into its
directory and start by configuring it::

  ./waf configure --mode=release --with-python --with-cpptests --with-examples --with-vamp

Use the keys:
   ``--with-python`` to enable python bindings,
   ``--with-examples`` to build examples based on the library,
   ``--with-vamp`` to build vamp plugin wrapper.

NOTE: you must *always* configure at least once before building!

The following will give you a list of options::

  ./waf --help

To compile everything you've configured::

  ./waf

To install the C++ library and the python bindings (if configured successfully; you might need to run this command with sudo)::

  ./waf install

To run the C++ base unit tests (only test basic library behavior)::

  ./waf run_tests

To run the python unit tests (include all unittests on algorithms, need python bindings installed first)::

  ./waf run_python_tests

To generate the full documentation (need python bindings installed first)::

  ./waf doc

Documentation will be located in ``doc/sphinxdoc/_build/html/`` folder.

All built examples (including the out-of-box features extractors) will be located in ``build/src/examples/`` folder, as well as the vamp plugin file ``libvamp_essentia.so``. In order to use the plugin you will need to place this file to the the standard vamp plugin folder of your system (such as ``/usr/local/lib/vamp/`` on Linux).


Using pre-trained high-level models in Essentia
-----------------------------------------------

The 2.0.1 version of Essentia includes a number of `pre-trained classifier models for genres, moods and instrumentation
<algorithms_overview.html#other-high-level-descriptors>`_. In order to use them you need to:

* Install Gaia2 library (supported on Linux/OSX): https://github.com/MTG/gaia/blob/master/README.md
* Build Essentia 2.0.1 with examples
* Use ``streaming_extractor_archivemusic``

