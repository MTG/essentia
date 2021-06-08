.. How-to install Essentia

Installing Essentia
===================

Mac OSX
-------
The easiest way to install Essentia on OSX is by using `our Homebrew formula <https://github.com/MTG/homebrew-essentia>`_. You will need to install `Homebrew package manager <http://brew.sh>`_ first (and there are other good reasons to do so apart from Essentia).


Note that packages location for Python installed via Homebrew is different from the system Python. If you plan to use Essentia with Python, make sure the Homebrew directory is at the top of your PATH environment variable. To this end, add the line::

  export PATH=/usr/local/bin:/usr/local/sbin:$PATH

at the bottom of your ``~/.bash_profile`` file. More information about using Python and Homebrew is `here <https://docs.brew.sh/Homebrew-and-Python>`_.


Linux
-----
You can install Essentia python extension from PyPi::

  pip install essentia

For other needs, you need to compile Essentia from source (see below).


Windows, Android, iOS
---------------------
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

- `Eigen <http://eigen.tuxfamily.org/>`_: for linear algebra
- `FFTW <http://www.fftw.org>`_: for the FFT implementation *(optional)*
- `libavcodec/libavformat/libavutil/libavresample <http://ffmpeg.org/>`_ (from the FFmpeg/LibAv project): for loading/saving any type of audio files *(optional)*
- `libsamplerate <http://www.mega-nerd.com/SRC/>`_: for resampling audio *(optional)*
- `TagLib <http://developer.kde.org/~wheeler/taglib.html>`_: for reading audio metadata tags *(optional)*
- `LibYAML <http://pyyaml.org/wiki/LibYAML>`_: for YAML files input/output *(optional)*
- `Gaia <https://github.com/MTG/gaia>`_: for using SVM classifier models *(optional)*
- `Chromaprint <https://github.com/acoustid/chromaprint>`_: for audio fingerprinting *(optional)*
- `TensorFlow <https://tensorflow.org>`_: for inference with TensorFlow deep learning models *(optional)*

All dependencies are optional, and some functionality will be excluded when a dependency is not found.

Installing dependencies on Linux
--------------------------------

You can install those dependencies on a Debian/Ubuntu system from official repositories using the command below::

  sudo apt-get install build-essential libeigen3-dev libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev libsamplerate0-dev libtag1-dev libchromaprint-dev

In order to use Python 3 bindings for the library, you might also need to install python3-dev, python3-numpy-dev (or python3-numpy on Ubuntu) and python3-yaml for YAML support in python::

  sudo apt-get install python3-dev python3-numpy-dev python3-numpy python3-yaml python3-six

Similarly, in the case of Python 2 install::

  sudo apt-get install python-dev python-numpy-dev python-numpy python-yaml python-six


Note that, depending on the version of Essentia, different versions of ``libav*`` and ``libtag1-dev`` packages are required. See `release notes for official releases <https://github.com/MTG/essentia/releases>`_.

Since the 2.1-beta3 release of Essentia, the required version of TagLib (``libtag1-dev``) is greater or equal to ``1.9``. The required version of LibAv (``libavcodec-dev``, ``libavformat-dev``, ``libavutil-dev`` and ``libavresample-dev``) is greater or equal to ``10``. The appropriate versions are distributed in Ubuntu 14.10 or later, and in Debian wheezy-backports. If you want to install Essentia on older versions of Ubuntu/Debian, you will have to `install a proper LibAv version from source <FAQ.html#build-essentia-on-ubuntu-14-04-or-earlier>`_.

If you are willing to use Essentia with a TensorFlow wrapper in C++, install the TensorFlow shared library using a helper script inside our source code::

  src/3rdparty/tensorflow/setup_from_libtensorflow.sh




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

  brew install eigen libyaml fftw ffmpeg libsamplerate libtag tensorflow

`Install python environment using Homebrew <http://docs.python-guide.org/en/latest/starting/install/osx>`_ (Note that you are advised to do as described here and there are `good reasons to do so <http://docs.python-guide.org/en/latest/starting/install/osx/>`_. You will most probably encounter installation errors when using python/numpy preinstalled with OSX.)::

  brew install python --framework
  pip install ipython numpy matplotlib pyyaml



Compiling Essentia
------------------

Once your dependencies are installed, you can proceed to compiling Essentia. Download Essentia's source code at `Github <https://github.com/MTG/essentia>`_.  Due to different dependencies requirement (see `release notes for official releases <https://github.com/MTG/essentia/releases>`_), make sure to download the version compatible with your system:

- **master** branch is the most updated version of Essentia in development
- **2.1 beta5** is the current stable version recommended to install.


Go into its source code directory and start by configuring it::

  ./waf configure --build-static --with-python --with-cpptests --with-examples --with-vamp

Use these (optional) flags:

- ``--with-python`` to enable python bindings,
- ``--with-examples`` to build `command line extractors <extractors_out_of_box.html>`_ based on the library,
- ``--with-vamp`` to build Vamp plugin wrapper,
- ``--with-gaia`` to build with Gaia library support,
- ``--with-tensorflow`` to build with TensorFlow support,
- ``--mode=debug`` to build in debug mode,
- ``--with-cpptests`` to build cpptests

NOTE: you must *always* configure at least once before building!

The following will give you the full list of options::

  ./waf --help

If you want to build with a custom toolchain, you can pass in the CC and CXX variables for using another compiler. For example, to build the library and examples with clang::

  CC=clang CXX=clang++ ./waf configure

To compile everything you've configured::

  ./waf

All built examples will be located in ``build/src/examples/`` folder, as well as the Vamp plugin file ``libvamp_essentia.so``.

To install the C++ library, python bindings, extractors and Vamp plugin (if configured successfully; you might need to run this command with sudo)::

  ./waf install



Compiling for Python 3
---------------------
The waf build script is a python script itself. By default it will configure Essentia to be built for the same Python that was used to execute this script. Alternatively, you can specify a specific Python binary to build for using the ``--python=PYTHON`` configuration option.

Therefore, to build for Python 3, you can either run all waf commands with your python3::

  python3 ./waf configure --build-static --with-python --with-cpptests --with-examples --with-vamp
  python3 ./waf
  python3 ./waf install

or specify the ``--python`` option, for example: ::

  ./waf configure --build-static --with-python --python=/usr/bin/python3 --with-cpptests --with-examples --with-vamp
  ./waf
  ./waf install

Note that when installing to ``/usr/local``, some Linux distributions provide a wrong default installation prefix
``/usr/local/lib/python3/dist-packages/`` for Python 3 extensions (for example, Ubuntu, see
`here <https://bugs.launchpad.net/ubuntu/+source/python3-defaults/+bug/1814653>`_ and
`here <https://bugs.launchpad.net/ubuntu/+source/python3-stdlib-extensions/+bug/1832215>`_).

To avoid import errors on these systems, specify the correct path in ``waf configure`` using a ``--pythondir`` flag or the ``PYTHONDIR``
environmental variable. For example, on Ubuntu 20.10 the correct path for the default Python 3.8 is ``/usr/local/lib/python3.8/dist-packages/``.

Alterantively, you can also configure the PYTHONPATH variable to include the `/usr/local/lib/python3/dist-packages/` path in the list of Python 3 `module search paths <https://docs.python.org/3/tutorial/modules.html#the-module-search-path>`_.

Finally, if you are having ``ImportError: libessentia.so: cannot open shared object file: No such file or directory`` in Python after installation on Linux, make sure that `/usr/local/lib` is included to `LD_LIBRARY_PATH` or run `ldconfig`.


Running tests (optional)
------------------------
If you want to assure that Essentia works correctly, do the tests. Some of the tests require additional audio files, which are stored in a separate submodule repository `essentia-audio <https://github.com/MTG/essentia-audio>`_. Make sure to clone Essentia git repository including its submodules in order to be able to run the tests (``git clone --recursive https://github.com/MTG/essentia.git``).

To run the C++ base unit tests (only test basic library behavior)::

  ./waf run_tests

To run the python unit tests (include all unittests on algorithms, need python bindings installed first)::

  ./waf run_python_tests

or, in the case if your default python is not Python 3::

  python3 ./waf run_python_tests


More information about running tests is `in our FAQ <FAQ.html#running-tests>`_.

Building documentation (optional)
---------------------------------

All documentation is provided on the official website of Essentia library. Follow the steps below to generate it by yourself.

Note that you should use Python 3 for building documentation. We rely on sphinxcontrib-doxylink which has already dropped support for Python2.

Install doxigen and pip3. If you are on Linux::

  sudo apt-get install doxygen python3-pip

Install additional dependencies (you might need to run this command with sudo)::

  sudo pip3 install sphinx pyparsing sphinxcontrib-doxylink docutils jupyter sphinxprettysearchresults
  sudo apt-get install pandoc

Make sure to build Essentia with Python 3 bindings and run::

  python3 ./waf doc

Documentation will be located in ``doc/sphinxdoc/_build/html/`` folder.


Building Essentia on Windows
----------------------------

Essentia C++ library and extractors based on it can be compiled and run correctly on Windows, but python bindings are not supported yet. The easiest way to build Essentia is by `cross-compilation on Linux using MinGW <FAQ.html#cross-compiling-for-windows-on-linux>`_. However the resulting library binaries are only compatible within C++ projects using MinGW compilers, and therefore they are not compatible with Visual Studio. If you want to use Visual Studio, there is no project readily available, so you will have to setup one yourself and compile the dependencies too.

Building Essentia in Windows Subsystem for Linux (WSL)
------------------------------------------------------
It is possible to install Essentia easily in the *Windows Subsystem for Linux* on Windows 10. This environment allows to run the same command-line utilities that could be run within your favorite `distribution <https://aka.ms/wslstore>`_. Note that WSL is still in its infancy and the methods of interoperability between Windows applications and WSL are likely to change in the future.

To install WSL, follow the `official guide <https://aka.ms/wsl2>`_.

After WSL is successfully installed, you should open a bash terminal and install the dependencies (see: `Installing dependencies on Linux`_).
Finally, you can compile Essentia (see: `Compiling Essentia`_).

Building Essentia on Android
----------------------------

A lightweight version of Essentia can be `cross-compiled for Android <FAQ.html#cross-compiling-for-android>`_ from Linux or Mac OSX.


Building Essentia on iOS
------------------------

A lightweight version of Essentia can be `cross-compiled for iOS <FAQ.html#cross-compiling-for-ios>`_ from Mac OSX.


Building Essentia for Web using asm.js or WebAssembly
-----------------------------------------------------

A lightweight version of Essentia can be cross-compiled to asm.js or WebAssembly targets using Emscripten for it's usage on the Web. See `FAQ <https://essentia.upf.edu/FAQ.htm>`_ for more details.

