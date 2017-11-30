Freesound Extractor
===================

``essentia_streaming_extractor_freesound`` is a configurable command-line feature extractor for sound analysis. It computes a large set of common sound descriptors, including various spectral, time-domain, rhythm and tonal characteristics, in an easy way with no programming required. It is designed for batch computations on large sound collections and is used by `Freesound <http://freesound.org/>`_. The prebuilt static binaries of this extractor are available via `Essentia website <http://essentia.upf.edu/documentation/extractors/>`_.

It is possible to customize the parameters of audio analysis, frame summarization, high-level classifier models, and output format, using a yaml profile file (`see below <freesound_extractor.html#configuration>`_). Writing your own custom profile file you can specify:

 - output format (json or yaml)
 - whether to store all frame values
 - an audio segment to analyze using time positions in seconds
 - analysis sample rate (audio will be converted to it before analysis, recommended and default value is 44100.0)
 - frame parameters for different groups of descriptors: frame/hop size, zero padding, window type (see `FrameCutter <reference/streaming_FrameCutter.html>`_ algorithm)
 - statistics to compute over frames: mean, var, median, min, max, dmean, dmean2, dvar, dvar2 (see `PoolAggregator <reference/streaming_PoolAggregator.html>`_ algorithm)


Sound Descriptors
-----------------
See below a detailed description of audio descriptors computed by the extractor. All descriptors are analyzed on a signal resampled to 44kHz sample rate (by default) and summed to mono. The frame-wise descriptors are `summarized <reference/std_PoolAggregator.html>`_ by their statistical distribution, but it is also possible to get frame values (disabled by default).


low-level.*
-----------

For implementation details, see `the code of extractor <https://github.com/MTG/essentia/blob/master/src/essentia/utils/extractor_freesound/FreesoundLowlevelDescriptors.cpp>`__.

By default all frame-based features are computed with frame/hop sizes equal to 2048/1024 samples unless stated otherwise.

* **sound_start_frame**, **sound_stop_frame**:  indices of frames at which sound begins and ends (all frames before and after are silent). Algorithms: `StartStopSilence <reference/streaming_StartStopSilence.html>`_

* **loudness_ebu128**: EBU R128 loudness descriptors. Algorithms: `LoudnessEBUR128 <reference/streaming_LoudnessEBUR128.html>`_

* **average_loudness**: dynamic range descriptor. It computes average loudness across frames, rescaled into the [0,1] interval. The value of 0 corresponds to signals with large dynamic range, 1 corresponds to signal with little dynamic range. Algorithms: `Loudness <reference/streaming_Loudness.html>`_

* **dynamic_complexity**: dynamic complexity computed on 2sec windows with 1sec overlap. Algorithms: `DynamicComplexity <reference/streaming_DynamicComplexity.html>`_

* **silence_rate_20dB**, **silence_rate_30dB**, **silence_rate_60dB**: rate of silent frames in a signal for thresholds of 20, 30, and 60 dBs. Algorithms: `SilenceRate <reference/streaming_SilenceRate.html>`_

* **spectral_rms**: spectral RMS. Algorithms: `RMS <reference/streaming_RMS.html>`_

* **spectral_flux**: spectral flux of a signal computed using L2-norm. Algorithms: `Flux <reference/streaming_Flux.html>`_

* **spectral_centroid**, **spectral_kurtosis**, **spectral_spread**, **spectral_skewness**: centroid and central moments statistics describing the spectral shape. Algorithms: `Centroid <reference/streaming_Centroid.html>`_, `CentralMoments <reference/streaming_CentralMoments.html>`_

* **spectral_rolloff**: the roll-off frequency of a spectrum. Algorithms: `RollOff <reference/streaming_RollOff.html>`_

* **spectral_decrease**: spectral decrease. Algorithms: `Decrease <reference/streaming_Decrease.html>`_

* **spectral_crest**: spectral crest. Algorithms: `Crest <reference/streaming_Crest.html>`_

* **spectral_flatness_db**: spectral flatness (dB): `FlatnessDB <reference/streaming_FlatnessDB.html>`_

* **hfc**: high frequency content descriptor as proposed by Masri. Algorithms: `HFC <reference/streaming_HFC.html>`_

* **spectral_strongpeak**: the Strong Peak of a signal’s spectrum. Algorithms: `StrongPeak <reference/streaming_StrongPeak.html>`_

* **zerocrossingrate** zero-crossing rate. Algorithms: `ZeroCrossingRate <reference/streaming_ZeroCrossingRate.html>`_

* **spectral_energy**: spectral energy. Algorithms: `Energy <reference/streaming_Energy.html>`_

* **spectral_energyband_low**, **spectral_energyband_middle_low**, **spectral_energyband_middle_high**, **spectral_energyband_high**: spectral energy in frequency bands [20Hz, 150Hz], [150Hz, 800Hz],  [800Hz, 4kHz], and [4kHz, 20kHz]. Algorithms `EnergyBand <reference/streaming_EnergyBand.html>`_

* **barkbands**: spectral energy in 27 Bark bands. Algorithms: `BarkBands <reference/streaming_BarkBands.html>`_

* **melbands**: spectral energy in 40 mel bands. Algorithms: `MFCC <reference/streaming_MFCC.html>`_

* **melbands96**: spectral energy in 96 mel bands. Algorithms: `MelBands <reference/streaming_MelBands.html>`_

* **erbbands**: spectral energy in 40 ERB bands. Algorithms: `ERBBands <reference/streaming_ERBBands.html>`_

* **mfcc**: the first 13 mel frequency cepstrum coefficients. See algorithm: `MFCC <reference/streaming_MFCC.html>`_

* **gfcc**: the first 13 gammatone feature cepstrum coefficients. Algorithms: `GFCC <reference/streaming_GFCC.html>`_

* **barkbands_crest**, **barkbands_flatness_db**: crest and flatness computed over energies in Bark bands. Algorithms: `Crest <reference/streaming_Crest.html>`_, `FlatnessDB <reference/streaming_FlatnessDB.html>`_

* **barkbands_kurtosis**, **barkbands_skewness**, **barkbands_spread**: central moments statistics over energies in Bark bands. Algorithms: `CentralMoments <reference/streaming_CentralMoments.html>`_

* **melbands_crest**, **melbands_flatness_db**:  crest and flatness computed over energies in mel bands. Algorithms: `Crest <reference/streaming_Crest.html>`_, `FlatnessDB <reference/streaming_FlatnessDB.html>`_

* **melbands_kurtosis**, **melbands_skewness**, **melbands_spread**:  central moments statistics over energies in mel bands. Algorithms: `CentralMoments <reference/streaming_CentralMoments.html>`_

* **erbbands_crest**, **erbbands_flatness_db**: crest and flatness computed over energies in ERB bands. Algorithms: `Crest <reference/streaming_Crest.html>`_, `FlatnessDB <reference/streaming_FlatnessDB.html>`_

* **erbbands_kurtosis**, **erbbands_skewness**, **erbbands_spread**: central moments statistics over energies in ERB bands. Algorithms: `CentralMoments <reference/streaming_CentralMoments.html>`_

* **dissonance**: sensory dissonance of a spectrum. Algorithms: `Dissonance <reference/streaming_Dissonance.html>`_

* **spectral_entropy**: Shannon entropy of a spectrum. Algorithms: `Entropy <reference/streaming_Entropy.html>`_

* **pitch_salience**: pitch salience of a spectrum. Algorithms: `PitchSalience <reference/streaming_PitchSalience.html>`_

* **pitch**, **pitch_instantaneous_confidence**: pitch estimation and its confidence (useful for monophonic sounds). Algorithms: `PitchYinFFT <reference/streaming_PitchYinFFT.html>`_

* **spectral_complexity**: spectral complexity. Algorithms: `SpectralComplexity <reference/streaming_SpectralComplexity.html>`_

* **spectral_contrast_coeffs**, **spectral_contrast_valleys**: spectral contrast features. Algorithms: `SpectralContrast <reference/streaming_SpectralContrast.html>`_


rhythm.*
--------

For implementation details, see `the code of extractor <https://github.com/MTG/essentia/blob/master/src/essentia/utils/extractor_freesound/FreesoundRhythmDescriptors.cpp>`__.

* **beats_position**: time positions [sec] of detected beats using beat tracking algorithm by Degara et al., 2012. Algorithms: `RhythmExtractor2013 <reference/streaming_RhythmExtractor2013.html>`_, `BeatTrackerDegara <reference/streaming_BeatTrackerDegara.html>`_

* **bpm_intervals**: time durations between consecutive beats

* **beats_count**: number of detected beats

* **bpm**: beats per minute (BPM) value according to detected beats

* **bpm_histogram**: BPM histogram. Algorithms: Algorithms: `BpmHistogramDescriptors <reference/streaming_BpmHistogramDescriptors.html>`_

* **bpm_histogram_first_peak_bpm**, **bpm_histogram_first_peak_spread**, **bpm_histogram_first_peak_weight**, **bpm_histogram_second_peak_bpm**, **bpm_histogram_second_peak_spread**, **bpm_histogram_second_peak_weight**: descriptors characterizing highest and second highest peak of the BPM histogram. Algorithms: `BpmHistogramDescriptors <reference/streaming_BpmHistogramDescriptors.html>`_

* **beats_loudness**, **beats_loudness_band_ratio**: spectral energy computed on beats segments of audio across the whole spectrum, and ratios of energy in 6 frequency bands. Algorithms: `BeatsLoudness <reference/streaming_BeatsLoudness.html>`_, `SingleBeatLoudness <reference/streaming_SingleBeatLoudness.html>`_

* **onset_times**, **onset_count**, **onset_rate** : time positions [sec] of detected onsets, its total number and rate per second. Algorithms: `OnsetRate <reference/streaming_OnsetRate.html>`_


tonal.*
-------

For implementation details, see `the code of extractor <https://github.com/MTG/essentia/blob/master/src/essentia/utils/extractor_freesound/FreesoundTonalDescriptors.cpp>`__. By default all features are computed with frame/hop sizes equal to 4096/2048 samples. 

* **tuning_frequency**: estimated tuning frequency [Hz]. Algorithms: `TuningFrequency <reference/streaming_TuningFrequency.html>`_

* **hpcp**, **thpcp**: 32-dimensional harmonic pitch class profile (HPCP) and its transposed version. Algorithms: `HPCP <reference/streaming_HPCP.html>`_

* **hpcp_entropy**: Shannon entropy of a HPCP vector. Algorithms: `Entropy <reference/streaming_Entropy.html>`_

* **hpcp_crest**: crest of the HPCP vector. Algorithms: `Crest <reference/streaming_Crest.html>`_

* **key_temperley**, **key_krumhansl**, **key_edma**; key estimation, its scale and strength using three different HPCP key profiles. Algorithms: `Key <reference/streaming_Key.html>`_

* **chords_progression**, **chords_count**, **chords_strength**, **chords_histogram**, **chords_changes_rate**, **chords_number_rate**, **chords_key**, **chords_scale**: estimated chords progression, number and strength of estimated chords, normalized histogram, chords change rate in the progression,  ratio of different chords from the total number of chords in the progression, key of the progression, taken as the most frequent chord, and scale of the progression, whether major or minor. Algorithms: `ChordsDetection <reference/streaming_ChordsDetection.html>`_, `ChordsDescriptors <reference/streaming_ChordsDescriptors.html>`_

* **tuning_diatonic_strength**: key strength estimated from high-resolution HPCP (120 dimensions) using diatonic profile. Algorithms: `Key <reference/streaming_Key.html>`_

* **tuning_equal_tempered_deviation**, **tuning_nontempered_energy_ratio**: equal-temperament deviation and non-tempered energy ratio estimated from high-resolution HPCP (120 dimensions). Algorithms: `HighResolutionFeatures <reference/streaming_HighResolutionFeatures.html>`_


sfx.*
-----

For implementation details, see `the code of extractor <https://github.com/MTG/essentia/blob/master/src/essentia/utils/extractor_freesound/FreesoundSfxDescriptors.cpp>`__.

Total and perceived sound duration:

* **duration**: total duration of an audio signal. Algorithms: `Duration <reference/streaming_Duration.html>`_.

* **effective_duration**: effective duration of the signal discarding silence (signal below the 10% of the envelope maximum). Algorithms: `Duration <reference/streaming_EffectiveDuration.html>`_.


Descriptors based on pitch and harmonics estimation:

* **oddtoevenharmonicenergyratio**: energy ratio between odd and even harmonics. Algorithms: `OddToEvenHarmonicEnergyRatio <reference/streaming_OddToEvenHarmonicEnergyRatio.html>`_.

* **tristimulus**: tristimulus.  Algorithms: `Tristimulus <reference/streaming_Tristimulus.html>`_.

* **inharmonicity**: inharmonisity. Algorithms: `Inharmonicity <reference/streaming_Inharmonicity.html>`_.


Sound envelope descriptors:

* **temporal_centroid**: ratio of the envelope centroid to total length. Algorithms: `Centroid <reference/streaming_Centroid.html>`_.

* **temporal_kurtosis**, **temporal_spread**, **temporal_skewness**: central moments statistics describing the signal envelope shape. Algorithms: `CentralMoments <reference/streaming_CentralMoments.html>`_

* **temporal_decrease**: signal envelope decrease. Algorithms: `Decrease <reference/streaming_Decrease.html>`_

* **tc_to_total**: ratio of the envelope centroid to total length. Algorithms: `TCToTotal <reference/streaming_TCToTotal.html>`_.

* **flatness**: the flatness coefficient of a signal envelope. Algorithms: `FlatnessSFX <reference/streaming_FlatnessSFX.html>`_.

* **logattacktime**:  the log10 of the attack time. Algorithms: `LogAttackTime <reference/streaming_LogAttackTime.html>`_.

* **max_to_total**: the maximum amplitude position to total envelope length ratio. Algorithms: `MaxToTotal <reference/streaming_MaxToTotal.html>`_.

* **tc_to_total**: the temporal centroid to total envelope length ratio. Algorithms: `TCToTotal <reference/streaming_TCToTotal.html>`_.

* **strongdecay**: the Strong Decay. Algorithms: `StrongDecay <reference/streaming_StrongDecay.html>`_.

* **der_av_after_max**: the average value of the envelope's derivative after the maximum amplitude position weighted by its amplitude (the smaller the value the more impulsive is a sound). Algorithms: `DerivativeSFX <reference/streaming_DerivativeSFX.html>`_.

* **max_der_before_max**:  the maximum value of the envelope's derivative before the maximum amplitude position (sounds with smooth attack phase will have lower values). Algorithms: `DerivativeSFX <reference/streaming_DerivativeSFX.html>`_.


Pitch envelope descriptors: 

* **pitch_centroid**: pitch envelope centroid. Algorithms: `Centroid <reference/streaming_Centroid.html>`_.
* **pitch_max_to_total**: ratio of the position of the maximum pitch value to total length. Algorithms: `MaxToTotal <reference/streaming_MaxToTotal.html>`_.
* **pitch_min_to_total**: ratio of the position of the minimum pitch value to total length. Algorithms: `MinToTotal <reference/streaming_MinToTotal.html>`_.
* **pitch_after_max_to_before_max_energy_ratio**: ratio of pitch envelope energy after the pitch maximum to pitch energy before the pitch maximum. Algorithms: `AfterMaxToBeforeMaxEnergyRatio <reference/streaming_AfterMaxToBeforeMaxEnergyRatio.html>`_.



Configuration
-------------

It is possible to customize the parameters of audio analysis, frame summarization, high-level classifier models, and output format, using a yaml profile file. Writing your own custom profile file you can:

Specify output format (json or yaml) ::

  outputFormat: json

Specify whether to store all frame values (0 or 1) ::

  outputFrames: 1

Specify an audio segment to analyze using time positions in seconds ::
  
  startTime: 0
  endTime: 10

Specify analysis sample rate (audio will be converted to it before analysis, recommended and default value is 44100.0) ::

  analysisSampleRate: 44100.0

Specify frame parameters for different groups of descriptors: frame/hop size, zero padding, window type (see `FrameCutter <reference/streaming_FrameCutter.html>`_ algorithm). Specify statistics to compute over frames: mean, var, median, min, max, dmean, dmean2, dvar, dvar2 (see `PoolAggregator <reference/streaming_PoolAggregator.html>`_ algorithm) ::

  lowlevel:
      frameSize: 2048
      hopSize: 1024
      zeroPadding: 0
      windowType: blackmanharris62
      silentFrames: noise
      stats: ["mean", "var", "median"]
  
  rhythm:
      method: degara
      minTempo: 40
      maxTempo: 208
      stats: ["mean", "var", "median", "min", "max"]

  tonal:  
      frameSize: 4096
      hopSize: 2048
      zeroPadding: 0
      windowType: blackmanharris62
      silentFrames: noise
      stats: ["mean", "var", "median", "min", "max"]

Specify whether you want to compute high-level descriptors based on classifier models associated with the respective filepaths (currently no models are provided out-of-box for sound classification, `see how to train your own models here <streaming_extractor_music.html#high-level-classifier-models>`_) ::

  highlevel:
      compute: 1
      svm_models: ['<path_to_gaia_svm_model1.history>', '<path_to_gaia_svm_model2.history>' ]


In the profile example below, the extractor is set to analyze only the first 30 seconds of audio and output frame values as well as their statistical summarization. ::

  startTime: 0
  endTime: 30
  outputFrames: 0
  outputFormat: json
  requireMbid: false
  indent: 4
  
  lowlevel:
      frameSize: 2048
      hopSize: 1024
      zeroPadding: 0
      windowType: blackmanharris62
      silentFrames: noise
      stats: ["mean", "var", "median", "min", "max", "dmean", "dmean2", "dvar", "dvar2"]
  
  rhythm:
      method: degara
      minTempo: 40
      maxTempo: 208
      stats: ["mean", "var", "median", "min", "max", "dmean", "dmean2", "dvar", "dvar2"]

  tonal:	
      frameSize: 4096
      hopSize: 2048
      zeroPadding: 0
      windowType: blackmanharris62
      silentFrames: noise
      stats: ["mean", "var", "median", "min", "max", "dmean", "dmean2", "dvar", "dvar2"]


.. |here| raw:: html

      <a
      href="http://htmlpreview.github.io/?https://github.com/MTG/essentia/blob/2.0.1/src/examples/svm_models/accuracies_2.0.1.html" target="_blank">here</a>
