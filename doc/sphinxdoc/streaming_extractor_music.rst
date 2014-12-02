Music extractor
===============

Usage
-----
``streaming_extractor_music`` computes a large set of spectral, time-domain, rhythm, tonal and high-level descriptors. The frame-wise descriptors are `summarized <reference/std_PoolAggregator.html>`_ by their statistical distribution. This extractor is suited for batch computations on large music collections and is used within `AcousticBrainz project <http://acousticbrainz.org/>`_. 

It is possible to customize the parameters of audio analysis, frame summarization, high-level classifier models, and output format, using a yaml profile file. For example, in the following profile, the extractor is set to analyze only the first 30 seconds of audio, output frame values as well as their statistical summarization, and apply two high-level models associated with the respective filepaths. ::

  startTime: 0
  endTime: 1e6
  analysisSampleRate: 44100.0
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
  
  average_loudness:
      frameSize: 88200
      hopSize: 44100
      windowType: hann
      silentFrames: noise

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

  highlevel:
      compute: 1
      svm_models: ['svm_models/genre_tzanetakis.history', 'svm_models/mood_sad.history' ]


High-level descriptors are `computed by classifier models <http://en.wikipedia.org/wiki/Statistical_classification>`_ from a lower-level representation of a music track in terms of summarized spectral, time-domain, rhythm, and tonal descriptors. Each model (a ``*.history`` file) is basically a `transformation history <reference/std_GaiaTransform.html>`_ that maps a pool (a `feature vector <http://en.wikipedia.org/wiki/Feature_vector>`_) of such lower-level descriptors produced by extractor into probability values of classes on which the model was trained. Due to algorithm improvements, different extractor versions may produce different descriptor values, uncompatible between each other. This implies that **the models you specify to use within the extractor have to be trained using the same version of the extractor to ensure consistency**. 
We provide such models pretrained on our ground truth music collections for each version of the music extractor via a `download page <http://essentia.upf.edu/documentation/svm_models/>`_.

Instead of computing high-level descriptors altogether with lower-level ones, it may be convenient to use ``streaming_extractor_music_svm``, a simplified extractor that computes high-level descriptors given a json/yaml file with spectral, time-domain, rhythm, and tonal descriptors required by classfier models (and produced by ``streaming_extractor_music``). High-level models are to be specified in a similar way via a profile file. ::

  highlevel:
      compute: 1
      svm_models: ['svm_models/genre_tzanetakis.history', 'svm_models/mood_sad.history' ]


Note, that you need to build Essentia with Gaia2 or use our static builds (soon online) in order to be able to run high-level models. Since Essentia version 2.1 high-level models are distributed apart from Essentia via a `download page <http://essentia.upf.edu/documentation/svm_models/>`_. 



Music descriptors
-----------------
See below a detailed description of provided audio descriptors. All descriptors are analyzed on a signal resampled to 44kHz sample rate, summed to mono and normalized using replay gain value. 


low-level.*
-----------

For implementation details, see `the code of extractor <https://github.com/MTG/essentia/blob/master/src/examples/extractor_music/MusicLowlevelDescriptors.cpp>`_.


The *spectral_centroid*, *spectral_kurtosis*, *spectral_spread*, *spectral_skewness*, *dissonance*, *spectral_entropy*, *spectral_contrast_coeffs*, and *spectral_contrast_valleys* are computed with an `equal-loudness filter <reference/streaming_EqualLoudness.html>`_ applied to the signal. All frame-based features are computed with frame/hop sizes equal to 2048/1024 samples unless stated otherwise.


* **average_loudness**: dynamic range descriptor. It rescales average loudness, computed on 2sec windows with 1 sec overlap, into the [0,1] interval. The value of 0 corresponds to signals with large dynamic range, 1 corresponds to signal with little dynamic range. Algorithms: `Loudness <reference/streaming_Loudness.html>`_

* **dynamic_complexity**: dynamic complexity computed on 2sec windows with 1sec overlap. Algorithms: `DynamicComplexity <reference/streaming_DynamicComplexity.html>`_

* **silence_rate_20dB**, **silence_rate_30dB**, **silence_rate_60dB**: rate of silent frames in a signal for thresholds of 20, 30, and 60 dBs. Algorithms: `SilenceRate <reference/streaming_SilenceRate.html>`_

* **spectral_rms**: spectral RMS. Algorithms: `RMS <reference/streaming_RMS.html>`_

* **spectral_flux**: spectral flux of a signal computed using L2-norm. Algorithms: `Flux <reference/streaming_Flux.html>`_

* **spectral_centroid**, spectral_kurtosis, spectral_spread, spectral_skewness: centroid and central moments statistics describing the spectral shape. Algorithms: `Centroid <reference/streaming_Centroid.html>`_, `CentralMoments <reference/streaming_CentralMoments.html>`_

* **spectral_rolloff**: the roll-off frequency of a spectrum. Algorithms: `RollOff <reference/streaming_RollOff.html>`_

* **spectral_decrease**: spectral decrease. Algorithms: `Decrease <reference/streaming_Decrease.html>`_

* **hfc**: high frequency content descriptor as proposed by Masri. Algorithms: `HFC <reference/streaming_HFC.html>`_

* **spectral_strongpeak**: the Strong Peak of a signal’s spectrum. Algorithms: `StrongPeak <reference/streaming_StrongPeak.html>`_

* **zerocrossingrate** zero-crossing rate. Algorithms: `ZeroCrossingRate <reference/streaming_ZeroCrossingRate.html>`_

* **spectral_energy**: spectral energy. Algorithms: `Energy <reference/streaming_Energy.html>`_

* **spectral_energyband_low**, **spectral_energyband_middle_low**, **spectral_energyband_middle_high**, **spectral_energyband_high**: spectral energy in frequency bands [20Hz, 150Hz], [150Hz, 800Hz],  [800Hz, 4kHz], and [4kHz, 20kHz]. Algorithms `EnergyBand <reference/streaming_EnergyBand.html>`_

* **barkbands**: spectral energy in 27 Bark bands. Algorithms: `BarkBands <reference/streaming_BarkBands.html>`_

* **melbands**: spectral energy in 40 mel bands. Algorithms: `MFCC <reference/streaming_MFCC.html>`_

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

* **spectral_complexity**: spectral complexity. Algorithms: `SpectralComplexity <reference/streaming_SpectralComplexity.html>`_

* **spectral_contrast_coeffs**, **spectral_contrast_valleys**: spectral contrast features. Algorithms: `SpectralContrast <reference/streaming_SpectralContrast.html>`_


rhythm.*
-----------

For implementation details, see `the code of extractor <https://github.com/MTG/essentia/blob/master/src/examples/extractor_music/MusicRhythmDescriptors.cpp>`_.

* **beats_position**: time positions [sec] of detected beats using beat tracking algorithm by Degara et al., 2012. Algorithms: `RhythmExtractor2013 <reference/streaming_RhythmExtractor2013.html>`_, `BeatTrackerDegara <reference/streaming_BeatTrackerDegara.html>`_

* **beats_count**: number of detected beats

* **bpm**: BPM value according to detected beats

* **bpm_histogram_first_peak_bpm**, **bpm_histogram_first_peak_spread**, **bpm_histogram_first_peak_weight**, **bpm_histogram_second_peak_bpm**, **bpm_histogram_second_peak_spread**, **bpm_histogram_second_peak_weight**: descriptors characterizing highest and second highest peak of the BPM probability histogram. Algorithms: `BpmHistogramDescriptors <reference/streaming_BpmHistogramDescriptors.html>`_

* **beats_loudness**, **beats_loudness_band_ratio**: spectral energy computed on beats segments of audio across the whole spectrum, and ratios of energy in 6 frequency bands. Algorithms: `BeatsLoudness <reference/streaming_BeatsLoudness.html>`_, `SingleBeatLoudness <reference/streaming_SingleBeatLoudness.html>`_

* **onset_rate**: number of detected onsets per second. Algorithms: `OnsetRate <reference/streaming_OnsetRate.html>`_

* **danceability**: danceability estimate. Algorithms: `Danceability <reference/streaming_Danceability.html>`_


tonal.*
-------

For implementation details, see `the code of extractor <https://github.com/MTG/essentia/blob/master/src/examples/extractor_music/MusicTonalDescriptors.cpp>`_. All features are computed with frame/hop sizes equal to 4096/2048 samples. 

* **tuning_frequency**: estimated tuning frequency [Hz]. Algorithms: `TuningFrequency <reference/streaming_TuningFrequency.html>`_

* **hpcp**, **thpcp**: 32-dimensional harmonic pitch class profile (HPCP) and its transposed version. Algorithms: `HPCP <reference/streaming_HPCP.html>`_

* **hpcp_entropy**: Shannon entropy of a HPCP vector. Algorithms: `Entropy <reference/streaming_Entropy.html>`_

* **key_key**, **key_scale**, **key_strength**: estimated key using Temperley’s profile, its scale and strength. Algorithms: `Key <reference/streaming_Key.html>`_

* **chords_strength**, **chords_histogram**, **chords_changes_rate**, **chords_number_rate**, **chords_key**, **chords_scale**: strength of estimated chords and normalized histogram of their progression; chords change rate in the progression;  ratio of different chords from the total number of chords in the progression; key of the progression, taken as the most frequent chord, and scale of the progression, whether major or minor. Algorithms: `ChordsDetection <reference/streaming_ChordsDetection.html>`_, `ChordsDescriptors <reference/streaming_ChordsDescriptors.html>`_

* **tuning_diatonic_strength**: key strength estimated from high-resolution HPCP (120 dimensions) using diatonic profile. Algorithms: `Key <reference/streaming_Key.html>`_

* **tuning_equal_tempered_deviation**, **tuning_nontempered_energy_ratio**: equal-temperament deviation and non-tempered energy ratio estimated from high-resolution HPCP (120 dimensions). Algorithms: `HighResolutionFeatures <reference/streaming_HighResolutionFeatures.html>`_


.. |here| raw:: html

      <a
      href="http://htmlpreview.github.io/?https://github.com/MTG/essentia/blob/2.0.1/src/examples/svm_models/accuracies_2.0.1.html" target="_blank">here</a>
