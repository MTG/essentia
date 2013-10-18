.. highlight:: python

Using executable extractors out-of-box
======================================

Introduction
------------

Although Essentia serves as a library, it also includes a number of examples that can be found in
the ``src/examples`` folder apart from its main code. Some of them are executables that can be used compute
sets of MIR descriptors given an audio file. When compiled they will be found in the folder 
``build/src/examples``.


Extractors
----------

These examples include several executable command-line feature extractors that you might use to familiarize
with the type of descriptors Essentia is able to compute or use them as a reference when building your own extractors:

* ``streaming_extractor_archivemusic``: computes a large set of spectral, time-domain, rhythm and tonal descriptors. 
  The frame-wise descriptors are summarized by their statistical distribution. This extractor is suited for batch 
  computations on large music collections.
* ``streaming_extractor``: a bit old extractor with a very cumbersome code (that should be rewritten) that 
  computes a large set of descriptors with a possibility of being parametrized given a profile file. 
  It can store results frame-wisely apart from statistical characterization, and apply segmentation. The 
  descriptor set is somewhat larger but less reliable than of ``streaming_extractor_archivemusic`` (includes 
  some unstable descriptors). 
* ``standard_pitchyinfft``: extracts pitch for a monophonic signal using `YinFFT <documentation/reference/std_PitchYinFFT.html>`_ algorithm.
* ``streaming_mfcc``: extracts MFCC frames and their statistical characterization.
* ``standard_rhythmtransform``: computes `rhythm transform <documentation/reference/std_RhythmTransform.html>`_.

Given an audio file these extractors produce a yaml or json file with results.

