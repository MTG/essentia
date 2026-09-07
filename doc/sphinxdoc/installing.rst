.. How-to install Essentia

Installing Essentia
===================

Installing from PyPI (wheels)
------------------------------

.. rubric:: What changed in this release

- Linux ``aarch64`` wheels are now published, for both ``essentia`` and
  ``essentia-tensorflow``, built on native arm64 runners.
- The macOS wheel floor is back down to ``macosx_15_0`` (it had drifted to
  ``macosx_26_2``). That was a side effect of the Homebrew ``libtensorflow`` bottle's own
  minimum OS version, not a property of TensorFlow itself; see `Installing TensorFlow`_
  below for where the library comes from now.
- The source distribution (sdist) builds cleanly again on Python 3.12 and newer: the
  3rdparty build script is now invoked explicitly with ``bash`` instead of relying on the
  execute bit surviving the sdist round-trip (`MTG/essentia#1462
  <https://github.com/MTG/essentia/issues/1462>`_).
- ``configure`` now requires FFmpeg/LibAv >= 5.1 and fails immediately, naming the version
  found and its ``.pc`` path, instead of failing deep inside a C++ compile; see `FFmpeg /
  LibAv version requirement`_ below.
- ``configure --with-tensorflow`` now link-tests the TensorFlow C API (``TF_Version``,
  ``TF_DeleteSession``) at configure time, instead of only failing later at import.
- The published x86_64 ``essentia`` wheels no longer carry a spurious ``libatomic.so.1``
  dependency that broke ``import essentia`` on images that don't ship ``libatomic1``, such
  as ``python:slim`` (`MTG/essentia#1541
  <https://github.com/MTG/essentia/issues/1541>`_).

Supported platforms
""""""""""""""""""""

Both ``essentia`` and ``essentia-tensorflow`` publish wheels for CPython 3.9 through 3.14
(free-threaded ``t`` builds are not built) on:

.. list-table::
   :header-rows: 1
   :widths: 16 20 30 34

   * - Platform
     - Wheel tag
     - ``essentia-tensorflow``'s TensorFlow C library
     - Notes
   * - Linux x86_64
     - ``manylinux2014`` (glibc 2.17)
     - Google's official tarball, TensorFlow 2.18.1
     -
   * - Linux aarch64
     - ``manylinux_2_28`` (glibc 2.28)
     - Homebrew ``libtensorflow`` 2.21.0 bottle
     - No gaia-backed algorithms (``GaiaTransform``, ``MusicExtractorSVM``): Gaia needs
       Qt 4.8, which has no AArch64 build.
   * - macOS arm64
     - ``macosx_15_0``
     - Homebrew ``libtensorflow`` 2.21.0 bottle (``arm64_sequoia``)
     - Built for macOS 15.0, the same floor as the Homebrew audio libraries the wheel also
       vendors (FFmpeg, TagLib, ...).
   * - macOS x86_64
     - ``macosx_15_0``
     - Google's official tarball, TensorFlow 2.16.2
     - 2.16.2 is the last darwin-x86_64 release Google published, and Homebrew no longer
       builds an Intel macOS bottle for ``libtensorflow``.

A source distribution (sdist) is also published, for any platform or Python version without
a prebuilt wheel; see `Installing from the sdist`_ below.

``essentia-tensorflow`` wheels are roughly 130-155 MB per platform: they vendor the pinned
TensorFlow C library above, so installing the package is the only thing a user does. There
is no ``tensorflow`` pip package to install alongside it, and none is declared as a
dependency. A TensorFlow build a user installs separately for their own code loads
independently, with no conflict.

``essentia`` vs. ``essentia-tensorflow``
"""""""""""""""""""""""""""""""""""""""""

``essentia-tensorflow`` is a superset of ``essentia``: the same package, built with
TensorFlow inference support compiled in. Install **exactly one of the two** -- both
install into the same ``essentia`` Python package name, so installing both in one
environment leaves whichever was installed second shadowing the first::

  pip install essentia              # no TensorFlow inference
  # or
  pip install essentia-tensorflow   # includes TensorFlow inference

There is no separate ``essentia.tensorflow`` module, and there never has been one. When
``essentia-tensorflow`` is the package installed, the TensorFlow algorithms
(``TensorflowPredict``, ``TensorflowPredictEffnetDiscogs``, ``TensorflowPredictMusiCNN``,
and the rest) simply appear alongside every other algorithm in ``essentia.standard`` and
``essentia.streaming``::

  from essentia.standard import TensorflowPredictEffnetDiscogs

See `Using machine learning models <machine_learning.html>`_ for how to run inference with
these algorithms.

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


Installing from the sdist
--------------------------
If no prebuilt wheel is available for your platform or Python version, pip falls back to
Essentia's source distribution (sdist) and builds it locally. This requires the same system
libraries as compiling from source (see `Installing dependencies on Linux`_ below)::

  sudo apt-get install build-essential libeigen3-dev libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libsamplerate0-dev libtag1-dev libchromaprint-dev

By default, building from the sdist also compiles static copies of these 3rdparty
dependencies. To use the system packages installed above instead, set
``ESSENTIA_WHEEL_SKIP_3RDPARTY=1``::

  ESSENTIA_WHEEL_SKIP_3RDPARTY=1 pip install --no-binary essentia essentia

Passing ``--no-binary essentia`` forces pip to build from the sdist even when a prebuilt wheel
exists, which is useful to verify that the source build works on your platform.


Windows, Android, iOS
---------------------
Cross-compile Essentia from Linux/macOS (see below).


Compiling Essentia from source
==============================

Essentia depends on (at least) the following libraries:

- `Eigen <http://eigen.tuxfamily.org/>`_: for linear algebra
- `FFTW <http://www.fftw.org>`_: for the FFT implementation *(optional)*
- `libavcodec/libavformat/libavutil/libswresample <http://ffmpeg.org/>`_ (from the FFmpeg/LibAv project), FFmpeg >= 5.1 or an equivalent LibAv release: for loading/saving any type of audio files *(optional)*
- `libsamplerate <https://libsndfile.github.io/libsamplerate/>`_: for resampling audio *(optional)*
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

Since the 2.1-beta3 release of Essentia, the required version of TagLib (``libtag1-dev``) is greater or equal to ``1.9``.

FFmpeg / LibAv version requirement
""""""""""""""""""""""""""""""""""

Essentia's audio loading/writing code uses the ``AVChannelLayout`` API, which FFmpeg
introduced in its 5.1 release. The ``configure`` step therefore requires, via pkg-config,
at least:

- ``libavcodec`` >= 59.37
- ``libavformat`` >= 59.27
- ``libavutil`` >= 57.28
- ``libswresample`` >= 4.7

These floors are declared once, as named constants, in ``src/wscript``. If an older
FFmpeg/LibAv is found on ``PKG_CONFIG_PATH``, ``configure`` fails immediately with the
found version, the ``.pc`` file it came from, and the minimum required version, rather
than letting the build fail deep inside a C++ compile on ``ch_layout``. ``configure`` also
compiles a small probe that uses ``AVChannelLayout`` directly, to guard against a
distribution reporting a satisfying version string without actually shipping that API. On
success, ``configure`` prints the selected ``libavcodec``/``libavformat``/``libavutil``/
``libswresample`` versions and their ``.pc`` file paths, so that when more than one FFmpeg
is installed, the one actually picked up via ``PKG_CONFIG_PATH`` is visible in the log.

If your distribution's FFmpeg/LibAv is older than this floor, `install a newer FFmpeg
version from source <FAQ.html#build-essentia-on-ubuntu-14-04-or-earlier>`_, or build one
under a separate prefix and point ``PKG_CONFIG_PATH`` (or ``./waf configure
--pkg-config-path``) at its ``pkgconfig`` directory.

If you want to use Essentia with a TensorFlow wrapper in C++, see `Installing TensorFlow`_ below.


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

  brew install eigen libyaml fftw ffmpeg libsamplerate libtag chromaprint

If you also want TensorFlow inference support, see `Installing TensorFlow`_ below for how
to provide the TensorFlow C library -- ``brew install libtensorflow``, not the much larger
``tensorflow`` formula.

`Install Python environment using Homebrew <http://docs.python-guide.org/en/latest/starting/install/osx>`_ (Note that you are advised to do as described here and there are `good reasons to do so <http://docs.python-guide.org/en/latest/starting/install/osx/>`_. You will most probably encounter installation errors when using Python/NumPy preinstalled with macOS.)::

  brew install python --framework
  pip install ipython numpy matplotlib pyyaml


Installing TensorFlow
---------------------

The ``TensorflowPredict*`` algorithms link against the TensorFlow C API. Essentia finds
it with pkg-config, so whatever you install has to provide a ``tensorflow.pc``, and
``configure --with-tensorflow`` link-tests it at configure time -- rather than leaving the
problem to be discovered later, at import -- failing if it cannot resolve ``TF_Version``
and ``TF_DeleteSession`` against the libraries that ``tensorflow.pc`` names. TensorFlow
2.13 is the minimum: from that release the C API is exported by ``libtensorflow_cc`` (pip
wheels) or ``libtensorflow`` (the standalone C library); earlier releases exported it only
from the Python wrapper extension, which cannot be linked against.

There are four ways to provide it:

Packaged libtensorflow
  The simplest option, where one exists. On macOS::

    brew install libtensorflow

  Some Linux distributions package it too; check for a ``libtensorflow`` or
  ``libtensorflow-dev`` package.

A pip TensorFlow wheel
  If your platform has no packaged ``libtensorflow``, generate a ``tensorflow.pc`` from a
  pip ``tensorflow`` wheel (2.13 or newer). The wheel carries the C API in
  ``libtensorflow_cc`` alongside a matching header tree, and the helper script points
  pkg-config at them in place rather than copying the library::

    pip3 install "tensorflow>=2.13"
    python3 src/3rdparty/tensorflow/setup_tensorflow.py --prefix ~/.local
    export PKG_CONFIG_PATH=~/.local/lib/pkgconfig:$PKG_CONFIG_PATH

  Linking against the wheel's libraries is also what lets a process import both
  ``essentia`` and ``tensorflow`` without the two loading separate copies of TensorFlow.
  The helper writes symlinks with linker-friendly names into ``<prefix>/lib`` and the
  ``tensorflow.pc`` itself into ``<prefix>/lib/pkgconfig``. Pick a prefix you own; the
  default of ``/usr/local`` needs ``sudo`` on most systems. Run
  ``setup_tensorflow.py --help`` for the remaining options, including ``--package-dir``
  to point at an unpacked wheel instead of an importable package.

The reproducible path the published wheels use
  ``packaging/fetch_libtensorflow.sh`` downloads and verifies the exact prebuilt
  TensorFlow C library each published ``essentia-tensorflow`` wheel is built against,
  pinned per platform in ``packaging/build_config.sh`` (see the `Supported platforms`_
  table above)::

    packaging/fetch_libtensorflow.sh --prefix ~/.local
    export PKG_CONFIG_PATH=~/.local/lib/pkgconfig:$PKG_CONFIG_PATH

  On linux/aarch64 and macOS arm64 these are Homebrew's ``libtensorflow`` 2.21.0 bottles,
  downloaded by digest from the GitHub Container Registry rather than installed with
  ``brew`` (so the pin does not move when Homebrew rebuilds). On macOS the script also
  rewrites the bottle's install names from Homebrew's placeholder to the prefix and re-signs
  the libraries, which is what ``brew`` does when it pours a bottle. On linux x86_64 and
  macOS x86_64 they are Google's official ``libtensorflow-cpu`` tarballs, at
  ``https://storage.googleapis.com/tensorflow/versions/<version>/libtensorflow-cpu-<platform>.tar.gz``:
  the linux x86_64 wheel is built on a glibc 2.17 image the bottle is too new for, and no
  Homebrew bottle exists for Intel macOS. Google only ever published that
  ``versions/<version>/`` layout for TensorFlow 2.16 through 2.18.1 -- the channel is frozen
  there, on every platform where it exists, and it never included a linux-arm64 build. The
  older ``.../tensorflow/libtensorflow/libtensorflow-cpu-<platform>-<version>.tar.gz`` path
  was retired at the same point and no longer resolves; don't probe it when checking whether
  a given platform/version has a tarball, it will falsely look unavailable. On macOS x86_64,
  2.16.2 is the newest version Google published a tarball for.

Your own ``tensorflow.pc``
  Write one and put its directory on ``PKG_CONFIG_PATH``. Its ``Cflags`` must make
  ``<tensorflow/c/c_api.h>`` resolve, and its ``Libs`` must name a library that exports the
  C API. ``libtensorflow_framework`` and the Python wrapper extension
  ``_pywrap_tensorflow_internal`` do not, on their own.


Building a wheel against a pip TensorFlow
"""""""""""""""""""""""""""""""""""""""""

This is a different, optional build path from the one above: the published
``essentia-tensorflow`` wheels do **not** use it. They vendor the pinned TensorFlow C
library from ``packaging/fetch_libtensorflow.sh`` (see above), which is what keeps
``essentia-tensorflow`` self-contained -- no ``tensorflow`` pip package is a runtime
dependency, and none is declared. What follows instead builds a wheel that loads
TensorFlow from a pip install next to it, for anyone packaging their own reduced-size
build. Skip it unless that is what you are doing.

A wheel built the ordinary way carries a copy of every library it links. For the
``essentia-tensorflow`` wheels that is a 600 MB copy of TensorFlow, roughly 90 per cent
of the download, for something the user very likely has installed already.
``--with-tensorflow-pip-rpath`` builds a wheel that leaves TensorFlow out and loads it
from the ``tensorflow`` package next door in ``site-packages`` instead::

  pip3 install "tensorflow>=2.13"
  python3 src/3rdparty/tensorflow/setup_tensorflow.py --prefix ~/.local
  export PKG_CONFIG_PATH=~/.local/lib/pkgconfig:$PKG_CONFIG_PATH
  python3 waf configure --with-python --with-tensorflow --with-tensorflow-pip-rpath

The flag adds rpath entries, and ``-Wl,-headerpad_max_install_names`` on macOS, to both
``libessentia`` and the Python extension. A relative one -- ``$ORIGIN/../tensorflow``,
or ``@loader_path/../tensorflow`` on macOS -- resolves TensorFlow from an installed
wheel, and is the entry that ships. An absolute one, to the TensorFlow package the build
linked against, exists only so that the wheel repair tools can *resolve* the libraries
they are told to exclude: exclusion happens after resolution, so ``delocate-wheel
--exclude`` fails with "Could not find all dependencies" without it. Both tools drop it
again on their way out. On ELF platforms a third entry, a plain ``$ORIGIN``, ends the
list; the comment on ``TENSORFLOW_PIP_RPATHS`` in ``src/wscript`` explains why it has to
be there.

Repair the wheel with TensorFlow excluded::

  # linux
  auditwheel repair --exclude libtensorflow_cc.so.2 \
      --exclude libtensorflow_framework.so.2 -w dist/ <wheel>
  # macOS
  delocate-wheel --exclude libtensorflow_cc --exclude libtensorflow_framework \
      -w dist/ <wheel>

``auditwheel`` keeps the relative entry on the extension and appends its own, giving
``$ORIGIN/../tensorflow:$ORIGIN:$ORIGIN/../<distribution>.libs``. It does *not* keep it
on the copy of ``libessentia`` it vendors, whose rpath it overwrites with ``$ORIGIN``
unconditionally; that copy still finds TensorFlow because the extension's entry is a
``DT_RPATH``, which the loader also searches for the libraries the extension pulls in.
``delocate`` keeps the entry on both.

The resulting wheel needs ``tensorflow`` at run time and should declare it as a
dependency: without it, ``import essentia.standard`` fails with an ``ImportError``
naming ``libtensorflow_cc``. No import-time shim is needed, and none should be added --
in particular a ``ctypes`` preload with ``RTLD_GLOBAL`` crashes the process.

Setting the environment variable ``ESSENTIA_TENSORFLOW_PIP_RPATH`` is equivalent to
passing the flag, and is the form to use when building wheels: ``setup.py`` runs
``waf configure`` a second time, for the bindings alone, and only the environment
reaches that invocation. Set it to ``1`` to have the build locate TensorFlow itself
(from ``tensorflow.pc``, or from the interpreter running the configure), or to the
``site-packages/tensorflow`` directory to name it outright, which is what a wheel build
wants because the interpreter that builds the bindings usually has no TensorFlow of its
own::

  export ESSENTIA_TENSORFLOW_PIP_RPATH=$(python3 -c 'import tensorflow, os.path;
      print(os.path.dirname(tensorflow.__file__))')

Neither the flag nor the variable is on by default: an ordinary source build links and
runs exactly as it did before.


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
- ``--with-tensorflow-pip-rpath`` to build a wheel that loads TensorFlow from a pip
  ``tensorflow`` install instead of vendoring it (see `Installing TensorFlow`_),
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

