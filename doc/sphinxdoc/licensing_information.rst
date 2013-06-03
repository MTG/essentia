Licensing Information
=====================

Some Essentia library algorithms use third-party libraries with licenses such as `GPL`_, `LGPL`_,
and others for which UPF is not able to provide full sublicensing rights to third parties.
If a third party plans to release a product based on Essentia, it needs to make sure to have
cleared the rights over the needed libraries for itself.

Essentia can interoperate with these libraries but it does not force any third-party
application to use them. UPF will not be liable for any illegal use of these libraries.

This is the set of libraries which can be used within essentia:

1. `GPL`_ (with dual license option to allow Commercial license. Commercial license has to
   be obtained from the software author)

   * FFTW - http://www.fftw.org/
   * Libsamplerate - http://www.mega-nerd.com/SRC/
   * MAD: MPEG Audio Decoder - http://www.underbit.com/products/mad/

2. `LGPL`_ (the original copyright notice as well as a copy of the LGPL has to be supplied.
   This can be achieved through dynamic linking)

   * Libsndfile - http://www.mega-nerd.com/libsndfile/
   * Taglib - http://developer.kde.org/~wheeler/taglib.html
   * Pthreads Win32 - http://sourceware.org/pthreads-win32/
   * XMLPP - http://sourceforge.net/projects/xmlpp/

3. Others (the original copyright notice must be retained. They can be linked dynamically or statically)

   * Libogg and Libvorbis - http://xiph.org/ogg/ (`BSD-style license`_)
   * BZ2 - http://www.bzip.org/ (`BSD-style license`_)
   * MersenneTwister - http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html (`BSD-style license`_)
   * Python - http://www.python.org/ (`Python license`_)
   * Intel Threading Building Blocks - http://www.threadingbuildingblocks.org/ (`GPL`_ with the `runtime exception`_)
   * Libxml2 - http://www.xmlsoft.org/ (`MIT license`_)
   * TNT - http://math.nist.gov/tnt/ (`Public domain`_)


.. _GPL: http://www.gnu.org/licenses/gpl.html
.. _LGPL: http://www.gnu.org/licenses/lgpl.html
.. _BSD-style license: http://www.opensource.org/licenses/bsd-license.php
.. _Python license: http://www.python.org/psf/license/
.. _runtime exception: http://gcc.gnu.org/onlinedocs/libstdc++/manual/bk01pt01ch01s02.html
.. _MIT license: http://www.opensource.org/licenses/mit-license.php
.. _Public domain: http://en.wikipedia.org/wiki/Public_domain
