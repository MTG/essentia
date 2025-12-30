.. How-to install Essentia

Installing Essentia
===================

macOS
-----
The easiest way to install Essentia on macOS is by using `our Homebrew formula <https://github.com/MTG/homebrew-essentia>`_. You will need to install `Homebrew package manager <http://brew.sh>`_ first (and there are other good reasons to do so apart from Essentia).


Note that packages location for Python installed via Homebrew is different from the system Python. If you plan to use Essentia with Python, make sure the Homebrew directory is at the top of your PATH environment variable. To this end, add the line::

  export PATH=/usr/local/bin:/usr/local/sbin:$PATH

at the bottom of your ``~/.bash_profile`` file. More information about using Python and Homebrew is `here <https://docs.brew.sh/Homebrew-and-Python>`_.


Linux
-----
You can install Essentia Python extension from PyPi::

  pip install essentia

For other needs, you need to compile Essentia from source (see below).


Windows, Android, iOS
---------------------
Cross-compile Essentia from Linux/macOS (see below).


Compiling Essentia from source
==============================

Essentia depends on (at least) the following libraries:

- `Eigen <http://eigen.tuxfamily.org/>`_: for linear algebra
- `FFTW <http://www.fftw.org>`_: for the FFT implementation *(optional)*
- `libavcodec/libavformat/libavutil/libswresample <http://ffmpeg.org/>`_ (from the FFmpeg/LibAv project): for loading/saving any type of audio files *(optional)*
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

  sudo apt-get install build-essential libeigen3-dev libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libsamplerate0-dev libtag1-dev libchromaprint-dev

In order to use Python 3 bindings for the library, you might also need to install python3-dev, python3-numpy-dev (or python3-numpy on Ubuntu) and python3-yaml for YAML support in python::

  sudo apt-get install python3-dev python3-numpy-dev python3-numpy python3-yaml python3-six

Note that, depending on the version of Essentia, different versions of ``libav*`` and ``libtag1-dev`` packages are required. See `release notes for official releases <https://github.com/MTG/essentia/releases>`_.

Since the 2.1-beta3 release of Essentia, the required version of TagLib (``libtag1-dev``) is greater or equal to ``1.9``. The required version of LibAv (``libavcodec-dev``, ``libavformat-dev``, ``libavutil-dev`` and ``libswresample-dev``) is greater or equal to ``10``. The appropriate versions are distributed in Ubuntu 14.10 or later, and in Debian wheezy-backports. If you want to install Essentia on older versions of Ubuntu/Debian, you will have to `install a proper LibAv version from source <FAQ.html#build-essentia-on-ubuntu-14-04-or-earlier>`_.

If you are willing to use Essentia with a TensorFlow wrapper in C++, install the TensorFlow shared library using a helper script inside our source code::

  src/3rdparty/tensorflow/setup_from_libtensorflow.sh




Installing dependencies on macOS
--------------------------------

Install Command Line Tools for Xcode. Even if you install Xcode from the app store you must configure command-line compilation by running::

  xcode-select --install

Install `Homebrew package manager <http://brew.sh>`_.

Insert the Homebrew directory at the top of your PATH environment variable by adding the following line at the bottom of your ``~/.profile`` file::

  export PATH=/usr/local/bin:/usr/local/sbin:$PATH

Install prerequisites::

  brew install pkg-config gcc readline sqlite gdbm freetype libpng

Install Essentia's dependencies::

  brew install eigen libyaml fftw ffmpeg@2.8 libsamplerate libtag chromaprint tensorflow

`Install Python environment using Homebrew <http://docs.python-guide.org/en/latest/starting/install/osx>`_ (Note that you are advised to do as described here and there are `good reasons to do so <http://docs.python-guide.org/en/latest/starting/install/osx/>`_. You will most probably encounter installation errors when using Python/NumPy preinstalled with macOS.)::

  brew install python --framework
  pip install ipython numpy matplotlib pyyaml



Compiling Essentia
------------------

Once your dependencies are installed, you can proceed to compiling Essentia. Download Essentia's source code at `Github <https://github.com/MTG/essentia>`_.  Due to different dependencies requirement (see `release notes for official releases <https://github.com/MTG/essentia/releases>`_), make sure to download the version compatible with your system:

- **master** branch is the most updated version of Essentia in development
- **2.1 beta5** is the current stable version recommended to install.


Go into its source code directory and start by configuring the build::

  python3 waf configure --build-static --with-python --with-cpptests --with-examples --with-vamp

Use these (optional) flags:

- ``--with-python`` to build with Python bindings,
- ``--with-examples`` to build `command line extractors <extractors_out_of_box.html>`_ based on the library,
- ``--with-vamp`` to build Vamp plugin wrapper,
- ``--with-gaia`` to build with Gaia support,
- ``--with-tensorflow`` to build with TensorFlow support,
- ``--mode=debug`` to build in debug mode,
- ``--with-cpptests`` to build cpptests

Note: you must *always* configure at least once before building!

The following will give you the full list of options::

  python3 waf --help

If you want to build with a custom toolchain, you can pass in the CC and CXX variables for using another compiler. For example, to build the library and examples with clang::

  CC=clang CXX=clang++ python3 waf configure

To compile everything you've configured::

  python3 waf

All built examples will be located in ``build/src/examples/`` folder, as well as the Vamp plugin file ``libvamp_essentia.so``.

To install the C++ library, Python bindings, extractors and Vamp plugin (if configured successfully; you might need to run this command with sudo)::

  python3 waf install


Python 3 bindings
-----------------
To build Essentia with Python 3 bindings, use the ``--with-python`` configuration flag.

By default, the waf build script will auto-detect the ``site-packages`` (or ``dist-packages``) directory to install Essentia's Python package according to the Python binary used to execute it. Alternatively, you can set a specific Python binary using the ``--python=PYTHON`` configuration option.

Note that when installing Essentia to the default ``/usr/local`` prefix, on some Linux distributions this results in a wrong ``/usr/local/lib/python3/dist-packages/`` package installation path (for example, Ubuntu, see
`here <https://bugs.launchpad.net/ubuntu/+source/python3-defaults/+bug/1814653>`_ and
`here <https://bugs.launchpad.net/ubuntu/+source/python3-stdlib-extensions/+bug/1832215>`_).

To avoid import errors on such systems, specify the correct path in ``waf configure`` using a ``--pythondir`` option or the ``PYTHONDIR`` environmental variable. For example, on Ubuntu 22.04 the correct path for the default Python 3.10 is ``/usr/local/lib/python3.10/dist-packages/``.

Alternatively, you can also configure the ``PYTHONPATH`` variable to include the ``/usr/local/lib/python3/dist-packages/`` path in the list of Python 3 `module search paths <https://docs.python.org/3/tutorial/modules.html#the-module-search-path>`_.

Finally, if you are having ``ImportError: libessentia.so: cannot open shared object file: No such file or directory`` in Python after installation on Linux, make sure that ``/usr/local/lib`` is included to ``LD_LIBRARY_PATH`` or run ``ldconfig``.


Running tests (optional)
------------------------
Run tests if you want to ensure that Essentia works correctly.

To run the C++ base unit tests (only test basic library behavior)::

  python3 waf run_tests

To run the Python unit tests (test all algorithms)::

  python3 waf run_python_tests

To run Python unit tests, you need to install Python bindings first. Some of these tests require additional audio files and binaries stored in `essentia-audio <https://github.com/MTG/essentia-audio>`_ and `essentia-models <https://github.com/MTG/essentia-models/>`_ submodule repositories. Therefore, make sure to clone Essentia git repository recursively with its submodules (``git clone --recursive https://github.com/MTG/essentia.git``).

Also, some tests require additional dependencies. Install those with::

  pip3 install scikit-learn

See more information about running tests `in our FAQ <FAQ.html#running-tests>`_.


Building documentation (optional)
---------------------------------

All documentation is provided on the official website of Essentia library. Follow the steps below to generate it by yourself.

Install doxigen and pip3. If you are on Linux::

  sudo apt-get install doxygen python3-pip

Install additional dependencies (you might need to run this command with sudo)::

  pip3 install sphinx pyparsing sphinxcontrib-doxylink docutils jupyter sphinx-toolbox nbformat gitpython
  sudo apt-get install pandoc

Make sure to build Essentia with Python 3 bindings and run::

  python3 waf doc

Documentation will be located in ``doc/sphinxdoc/_build/html/`` folder.

Note: Code examples embedded in the documentation page for Essentia Models require Python example files located in ``src/examples/python/models/scripts/``. These scripts can be automatically regenerated with ``src/examples/python/models/generate_example_scripts.sh``.




Building Essentia on Windows
----------------------------

Essentia C++ library and extractors based on it can be compiled and run correctly on Windows, but Python bindings are not supported yet. The easiest way to build Essentia is by `cross-compilation on Linux using MinGW <FAQ.html#cross-compiling-for-windows-on-linux>`_. However the resulting library binaries are only compatible within C++ projects using MinGW compilers, and therefore they are not compatible with Visual Studio. If you want to use Visual Studio, there is no project readily available, so you will have to setup one yourself and compile the dependencies too.

Building Essentia in Windows Subsystem for Linux (WSL)
------------------------------------------------------
It is possible to install Essentia easily in the *Windows Subsystem for Linux* on Windows 10. This environment allows to run the same command-line utilities that could be run within your favorite `distribution <https://aka.ms/wslstore>`_. Note that WSL is still in its infancy and the methods of interoperability between Windows applications and WSL are likely to change in the future.

To install WSL, follow the `official guide <https://aka.ms/wsl2>`_.

After WSL is successfully installed, you should open a bash terminal and install the dependencies (see: `Installing dependencies on Linux`_).
Finally, you can compile Essentia (see: `Compiling Essentia`_).

Building Essentia on Android
----------------------------

A lightweight version of Essentia can be `cross-compiled for Android <FAQ.html#cross-compiling-for-android>`_ from Linux or macOS.


Building Essentia on iOS
------------------------

A lightweight version of Essentia can be `cross-compiled for iOS <FAQ.html#cross-compiling-for-ios>`_ from macOS.


Building Essentia for Web using asm.js or WebAssembly
-----------------------------------------------------

A lightweight version of Essentia can be cross-compiled to asm.js or WebAssembly targets using Emscripten for it's usage on the Web. See `FAQ <https://essentia.upf.edu/FAQ.html>`_ for more details.

