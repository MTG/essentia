essentia
========

Essentia is an open-source C++ library for audio analysis and audio-based music information retrieval released under
the Affero GPL license. It contains an extensive collection of reusable algorithms which implement audio input/output
functionality, standard digital signal processing blocks, statistical characterization of data, and a large set of
spectral, temporal, tonal and high-level music descriptors. The library is also wrapped in Python and includes a number
of predefined executable extractors for the available music descriptors, which facilitates its use for fast prototyping
and allows setting up research experiments very rapidly. Furthermore, it includes a Vamp plugin to be used with
Sonic Visualiser for visualization purposes. The library is cross-platform and currently supports Linux, Mac OS X,
and Windows systems. Essentia is designed with a focus on the robustness of the provided music descriptors and is
optimized in terms of the computational cost of the algorithms. The provided functionality, specifically the music
descriptors included in-the-box and signal processing algorithms, is easily expandable and allows for both research
experiments and development of large-scale industrial applications.

Documentation online: http://essentia.upf.edu/

Installation HOWTO: http://essentia.upf.edu/documentation/installing.html or read the doc/sphinxdoc/installing.rst file
in the code. 

Quick start using python: http://essentia.upf.edu/documentation/python_tutorial.html

Essentia does compile and run correctly on Windows, however there is no Visual Studio project readily available, so you
will have to setup one yourself and compile the dependencies too. We will be working on Windows installer in the near
future.

