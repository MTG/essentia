
Algorithms overview
===================

This is a slightly detailed list describing the main algorithms you can find in Essentia
as well as a small description.

Please note that this is not an exhaustive list but rather a quick overview of the most used
algorithms in Essentia. For the full list of algorithms and their detailed documentation,
please see the `complete reference <algorithms_reference.html>`_.


Audio input / output
--------------------

Essentia has a variety of audio loaders to provide a very convenient way to load audio files from disk.

* ``AudioLoader``: this is the most basic audio loader. It will load the given audio file and
  return the stream of stereo samples. If the audio file was mono, only the left channel is used.
  It will load virtually any format existing as it uses the `FFmpeg`_ libraries to load audio.
  It will also load the audio from a given video file (works even with Flash files!).
* ``MonoLoader``: probably the most used audio loader, the ``MonoLoader`` will downmix the file
  to a mono signal and resample it to the given sample rate (default 44.1 KHz).
* ``EasyLoader``: a ``MonoLoader`` which can also trim the audio to a given start/end time and
  normalize the resulting samples using a given `ReplayGain`_ value.
* ``EqloudLoader``: an ``EasyLoader`` that applies an `Equal-loudness`_ filter to the resulting
  audio
* ``MetadataReader``: reads the metadata tags stored in the given file (e.g.: mp3 tags, ...).

Essentia can also write audio files, using the following algorithms:

* ``AudioWriter``: a versatile audio writer that can write a stream of samples to any format
  supported by `FFmpeg`_. Takes as input a stream of stereo samples.
* ``MonoWriter``: an ``AudioWriter`` that takes as input a stream of mono samples.
* ``AudioOnsetsMarker``: will write an audio file with the given stream of samples, on which
  beeps will have been added at the times of the given onsets.


Standard signal processing algorithms
-------------------------------------

The following algorithms are quite standard in any signal processing library:

* ``FrameCutter``: takes an input audio stream and cuts it into frames.
* ``Windowing``: returns a windowed frame (knows a lot of standard window types, such as Hann,
  Hamming, Blackman-Harris, ...).
* ``FFT``: the omnipresent (complex) `Fast Fourier Transform`_ algorithm.
* ``Spectrum``: returns only the magnitude part of the ``FFT`` algorithm.
* ``DCT``: the type II `Discrete Cosine Transform`_.
* ``Envelope``: an envelope follower.
* ``PeakDetection``: to detect peaks in an array of values.
* ``Resample``: to resample a given audio signal.
* ``Slicer``: returns the given slices (start/end times) of an audio stream.
* ``ReplayGain``: returns the `ReplayGain`_ value for the given audio.
* ``CrossCorrelation``: computes the `cross-correlation`_ of two signals.
* ``AutoCorrelation``: computes the `auto-correlation`_ of a signal.
* ``CartesianToPolar`` and ``PolarToCartesian``: do the conversion of complex arrays between
  the polar and the cartesian coordinate systems.

Statistics
----------

The following algorithms compute statistics over an array of values, or do some kind
of aggregation:

* ``Mean``, ``GeometricMean`` and ``PowerMean``: do what their names hint at.
* ``Energy``: computes the energy of the array.
* ``RMS``: computes the Root mean square of the array.
* ``SingleGaussian``: computes a single Gaussian estimation for the given list of arrays (returns the
  mean array, its covariance and inverse covariance matrices)
* ``CentralMoments`` and ``RawMoments``: compute the moments up to the 5th-order of the array.
* ``Variance``, ``Skewness`` and ``Kurtosis``: are often used to describe probability distributions.
* ``DistributionShape``: combines the ``Variance``, ``Skewness`` and ``Kurtosis`` values in a single vector.
* ``Median``, ``Flatness``, ``Crest``, ``InstantPower`` are some others lesser used algorithms.

Filters
-------

Essentia implements a nice variety of audio filters:

* ``IIR``: a generic algorithm that does `IIR`_ filtering, where you can specify the coefficients manually.
* ``LowPass``, ``BandPass`` and ``HighPass``: do 1st or 2nd order low-, band- and high-pass filtering.
* ``BandPass``, ``BandReject``: do band-pass and band-rejection filtering.
* ``DCRemoval``: removes the DC component of a signal.
* ``EqualLoudness``: filters the signal using an equal-loudness curve approximating filter.

MIR low-level descriptors
-------------------------

The following algorithms are either spectral or time based, and are used quite often as descriptors in
the `Music Information Retrieval`_ community.

Spectral descriptors
^^^^^^^^^^^^^^^^^^^^

* ``BarkBands``: computes the `Bark band <http://en.wikipedia.org/wiki/Bark_scale>`_ energies.
* ``MelBands``: computes the `Mel band <http://en.wikipedia.org/wiki/Mel_scale>`_ energies.
* ``MFCC``: computes the `Mel-frequency cepstral coefficients
  <http://en.wikipedia.org/wiki/Mel-frequency_cepstral_coefficient>`_ of a frame.
* ``HFC``: computes the `High-Frequency Content <http://en.wikipedia.org/wiki/High_Frequency_Content_measure>`_ measure.
* ``HPCP``: computes the Harmonic Pitch-Class Profile of a spectrum (also called Chroma features).
* ``SpectralContrast``: is used for music genre classification.
* ``SpectralWhitening``: whitens the input spectrum.
* ``Panning``: computes the `panorama distribution <http://en.wikipedia.org/wiki/Panning_(audio)>`_
  of a stereo audio frame.

Temporal descriptors
^^^^^^^^^^^^^^^^^^^^

* ``Duration`` and ``EffectiveDuration``: compute the total duration of a signal and the duration
  of the signal being above a certain energy level.
* ``LPC``: computes the `Linear Predictive Coding <http://en.wikipedia.org/wiki/Linear_predictive_coding>`_
  coefficients of a frame as well as the associated reflection coefficients.
* ``ZCR``: computes the `Zero-crossing rate <http://en.wikipedia.org/wiki/ZCR>`_ of the signal.
* ``Leq``, ``LARM``, ``Loudness`` and ``LoudnessVicker``: are all loudness measures.



Tonal descriptors
-----------------

The following algorithms are related to computing a tonal representation of the audio signal:

* ``PitchDetection``: computes the pitch for an audio frame.
* ``TuningFrequency``: returns the exact frequency on which a song is tuned and the number of cents to 440Hz.
* ``Key``: returns the key and scale of a song.
* ``ChordsDetection``: computes the sequence of chords in a song.
* ``ChordsDescriptors``: computes some descriptors associated with the sequence of chords, such as its histogram, etc.
* ``Inharmonicity`` and ``Dissonance``: both try to estimate whether an audio frame "sounds" harmonic or not.


Rhythmic descriptors
--------------------

The following algorithms are related to computing a rhythmic representation of the audio signal:

* ``RhythmTransform``: computes a rhythmical representation, similar to an FFT but in the rhythm domain.
* ``TempoTap``: is Essentia's first BPM extractor.
* ``RhythmDescriptors``: computes some statistics about the tempogram of a song.
* ``BeatsLoudness``: computes the loudness of the signal on windows centered around the beat locations.


SFX descriptors
---------------

Most of the algorithms are designed to deal with music audio files, but the following ones are
intended to be used with short sounds instead of full-length music tracks:

* ``LogAttackTime``: returns the logarithm of the attack time for the sound.
* ``MaxToTotal`` and ``MinToTotal``: return a measure of whether the max (resp. min) value of
  a sound is located towards its beginning or end.
* ``PitchSalience``: returns whether the pitch is salient (ie: strongly marked) in a sound.
* ``TCToTotal``: computes the normalized position of the temporal centroid.


High-level descriptors
----------------------

Essentia also contains the following mid- and high-level descriptors:

* ``Danceability``: returns whether a song is "danceable".
* ``DynamicComplexity``: returns whether a song has a high dynamic range.
* ``FadeDetection``: detects fade-ins/fade-outs in a song.
* ``SBic``: segments a song using the Bayesian Information Criterion.
* ``NoveltyCurve``: computes the novelty curve for the audio signal.
* ``Onsets``: computes the list of onsets in thge audio signal.
* ``PCA``: does the `Principal Component Analysis <http://en.wikipedia.org/wiki/Principal_component_analysis>`_
  of the given list of arrays.
* ``GaiaTransform``: applies the given Gaia transformation to a Pool. This is mostly used for
  classifiers which have been trained with Gaia. At the moment, the following classifiers have
  trained models available in Essentia:

  * musical genre (4 different databases)
  * ballroom music classification
  * moods: happy, sad, aggressive, relaxed, also by extension: acoustic, electronic, party
  * western / non-western music
  * live / studio recording
  * perceptual speed (ie: slow, medium, fast)
  * voice / instrumental
  * gender (male / female singer)
  * timbre classification
  * speech / music


Extractors
----------

As Essentia algorithms can themselves be composed of multiple algorithms
(see :doc:`Composite algorithms <composite_api>`),
a few useful extractors have been written as algorithms. They are the following:

* ``LevelExtractor``: computes the loudness of a music track.
* ``LowLevelSpectralExtractor``: computes a lot of low-level features from a music stream.
* ``LowLevelSpectralEqloudExtractor``: computes a lot of low-level features from a music
  stream which should have been previously equal-loudness filtered.
* ``RhythmExtractor``: computes the BPM of a song as well as other rhythm-related features
  using TempoTap.
* ``TuningFrequencyExtractor``: computes the tuning frequency of a music track.
* ``KeyExtractor``: computes the key and scale of a music track.
* ``TonalExtractor``: computes the tonal information of a music track (key, scale, chords
  sequence, chords histogram, ...)
* ``Extractor``: extracts pretty much all the features useful as descriptors for doing music similarity.



What next?
----------

For more information about ``Algorithms``, please see the `complete reference <algorithms_reference.html>`_.

For information on the other types and classes of Essentia which are not ``Algorithms``, see the `Design Overview <design_overview.html>`_ page.

For a tutorial showing how to use these algorithms in practice, read the tutorial for either `python <python_tutorial.html>`_ or `C++ <howto_standard_extractor.html>`_.

For more advanced examples, you can also look at the `src/examples`_ directory of Essentia's git repository.

.. _src/examples: https://github.com/MTG/essentia/tree/master/src/examples
.. _FFmpeg: http://www.ffmpeg.org/
.. _ReplayGain: http://www.replaygain.org/
.. _Equal-loudness: http://replaygain.hydrogenaudio.org/equal_loudness.html
.. _Fast Fourier Transform: http://en.wikipedia.org/wiki/Fft
.. _cross-correlation: http://en.wikipedia.org/wiki/Cross-correlation
.. _auto-correlation: http://en.wikipedia.org/wiki/Autocorrelation
.. _Root mean square: http://en.wikipedia.org/wiki/Root_mean_square
.. _Discrete Cosine Transform: http://en.wikipedia.org/wiki/Discrete_cosine_transform
.. _IIR: http://en.wikipedia.org/wiki/IIR
.. _Music Information Retrieval: http://en.wikipedia.org/wiki/Music_information_retrieval
