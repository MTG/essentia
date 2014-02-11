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

* ``streaming_extractor_archivemusic``: computes a large set of spectral, time-domain, rhythm, tonal and high-level descriptors. 
  The frame-wise descriptors are summarized by their statistical distribution. This extractor is suited for batch 
  computations on large music collections. 
  
    Note, that you need to compile Essentia 2.0.1 with Gaia2 if you want to include high-level models. The accuracies of these models are presented |here|.
  

* ``streaming_extractor_freesound``: similar extractor recommended for sound analysis. This extractor is used by `Freesound <http://freesound.org>`_ in order to provide sound analysis API and search by similar sounds functionality.

* ``streaming_extractor``: a bit old extractor with a very cumbersome code (that should be rewritten) that 
  computes a large set of descriptors with a possibility of being parametrized given a profile file. 
  It can store results frame-wisely apart from statistical characterization, and apply segmentation. The 
  descriptor set is somewhat larger but less reliable than of ``streaming_extractor_archivemusic`` (includes 
  some unstable descriptors). 

* ``standard_pitchyinfft``: extracts pitch for a monophonic signal using `YinFFT <reference/std_PitchYinFFT.html>`_ algorithm.

* ``streaming_predominantmelody``: extracts pitch of a predominant melody using `MELODIA <reference/std_PredominantMelody.html>`_ algorithm. 

* ``streaming_mfcc``: extracts MFCC frames and their statistical characterization.

* ``standard_rhythmtransform``: computes `rhythm transform <reference/std_RhythmTransform.html>`_.

Given an audio file these extractors produce a yaml or json file with results.

.. |here| raw:: html

      <a
      href="http://htmlpreview.github.io/?https://github.com/MTG/essentia/blob/2.0.1/src/examples/svm_models/accuracies_2.0.1.html" target="_blank">here</a>
