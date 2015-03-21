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

* ``streaming_extractor_music``: computes a large set of spectral, time-domain, rhythm, tonal and high-level descriptors. 
  The frame-wise descriptors are summarized by their statistical distribution. This extractor was designed for batch computations on large music collections. See `detailed documentation <streaming_extractor_music.html>`_.

 

* ``streaming_extractor_freesound``: similar extractor recommended for sound analysis. This extractor is used by `Freesound <http://freesound.org>`_ in order to provide sound analysis API and search by similar sounds functionality.

* ``streaming_extractor``: outdated extractor with a large set of descriptors and segmentation. The 
  descriptor set include some unstable descriptors and is less reliable than of ``streaming_extractor_music``.

* ``standard_pitchyinfft``: extracts pitch for a monophonic signal using `YinFFT <reference/std_PitchYinFFT.html>`_ algorithm.

* ``streaming_predominantmelody``: extracts pitch of a predominant melody using `MELODIA <reference/std_PredominantMelody.html>`_ algorithm. 

* ``streaming_beattracker_multifeature_mirex2013``: extracts beat postions usign the `multifeature beattracker <reference/std_BeatTrackerMultiFeature.html>`_ algorithm.

* ``streaming_mfcc``: extracts MFCC frames and their statistical characterization.

* ``standard_rhythmtransform``: computes `rhythm transform <reference/std_RhythmTransform.html>`_.

Given an audio file these extractors produce a yaml or json file with results.
