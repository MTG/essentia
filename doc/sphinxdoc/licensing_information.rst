Licensing Essentia
==================

Essentia is the result of more than 10 years of research and development at Music Technology Group and it 
incorporates the work done by many researchers during all this time. The library has been designed to support 
large-scale industrial applications. It is being used to support applications such as music and sound classification, 
semantic auto-tagging, music similarity and recommendation, visualization and interaction with music, sound 
indexing, musical instruments detection, cover detection, and beat detection among many others.

Essentia is available under an open licence, `Affero GPLv3 <http://www.gnu.org/licenses/agpl.html>`_, 
for non-commercial applications, thus it is possible to test the library before deciding to licence 
it under a comercial licence.

Contact `Music Technology Group (UPF) <https://www.upf.edu/web/mtg/technologies-licensing>`_ for 
more information about licensing conditions and for consulting how Essentia can suit your application.


Licensing dependencies
======================

Some Essentia library algorithms use third-party libraries with licenses such as `GPL`_, `LGPL`_,
and others for which UPF is not able to provide full sublicensing rights to third parties.
If a third party plans to release a product based on Essentia, it needs to make sure to have
cleared the rights over the required libraries for itself.

Essentia can interoperate with these libraries but it does not force any third-party
application to use them. It is possible to `compile Essentia with reduced dependencies <http://essentia.upf.edu/documentation/FAQ.html#building-lightweight-essentia-with-reduced-dependencies>`_ ignoring undesired 3rd-party libraries and algorithms. 
UPF will not be liable for any illegal use of these libraries.


This is the set of libraries which can be used within Essentia:

1. `GPL`_ (with dual license option to allow Commercial license. Commercial license has to
   be obtained from the software author)

   * FFTW - http://www.fftw.org/ -- used by FFT and IFFT algorithms, it can be replaced by more permissive `Kiss FFT or Accelerate <http://essentia.upf.edu/documentation/FAQ.html#building-lightweight-essentia-with-reduced-dependencies>`_
   * NNLS Chroma - http://www.isophonics.net/nnls-chroma -- used by LogSpectrum and NNLSChroma algorithms. Alternatively, Essentia's ConstantQ and HPCP native algorithms should provide equilavent results.

2. `LGPL`_ (the original copyright notice as well as a copy of the LGPL has to be supplied
   This can be achieved through dynamic linking)
   
   * libavcodec/libavformat/libavutil/libavresample - https://www.ffmpeg.org - used by the AudioLoader, MonoLoader, EasyLoader and EqloudLoader algorithms
   * Taglib - http://developer.kde.org/~wheeler/taglib.html - used by the MetadataReader algorithm
   * Chromaprint - https://github.com/acoustid/chromaprint - used by the Chromaprinter algorithm
   * splineutil.hpp and splineutil.cpp - http://people.sc.fsu.edu/~jburkardt/cpp_src/spline/spline.html - used by the Spline and CubicSpline algorithms

3. Others (the original copyright notice must be retained. They can be linked dynamically or statically)

   * Libsamplerate - http://www.mega-nerd.com/SRC/ - used by the Resample algorithm, and MonoLoader, EasyLoader and EqloudLoader. Originally released under GPL, it is licensed under `BSD-2-Clause`_ since 2016
   * MersenneTwister - http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html (`BSD-style license`_)
   * TNT - http://math.nist.gov/tnt/ (`Public domain`_)
   * VAMP SDK - http://www.vamp-plugins.org/develop.html (`MIT license`_) (for building vamp plugins)
   * Python - http://www.python.org/ (`Python license`_)
   * NumPy - http://www.numpy.org/
   * LibYAML - http://pyyaml.org/wiki/LibYAML (`MIT license`_)
   * Kiss FFT - https://sourceforge.net/projects/kissfft/ (`BSD-style license`_)
   * Accelerate - https://developer.apple.com/reference/accelerate


.. _GPL: http://www.gnu.org/licenses/gpl.html
.. _LGPL: http://www.gnu.org/licenses/lgpl.html
.. _BSD-style license: http://www.opensource.org/licenses/bsd-license.php
.. _Python license: http://www.python.org/psf/license/
.. _runtime exception: http://gcc.gnu.org/onlinedocs/libstdc++/manual/bk01pt01ch01s02.html
.. _MIT license: http://www.opensource.org/licenses/mit-license.php
.. _Public domain: http://en.wikipedia.org/wiki/Public_domain
.. _BSD-2-Clause: https://opensource.org/licenses/BSD-2-Clause

