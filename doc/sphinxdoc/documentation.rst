Essentia
========

.. toctree::
   :maxdepth: 1
   :titlesonly:



``essentia`` is an open-source C++ library with Python and JavaScript bindings for **audio analysis and audio-based music information retrieval**. 
It is released under the `Affero GPLv3 license <https://www.tldrlegal.com/license/gnu-affero-general-public-license-v3-agpl-3-0>`_ and is also available under a proprietary license :doc:`upon request <licensing_information>`.
The following algorithms are included:

* **Audio file input/output**: ability to read and write nearly all audio file formats (wav, mp3, ogg, flac, etc.)
* **Standard signal processing blocks**: FFT, DCT, frame cutter, windowing, envelope, smoothing
* **Filters (FIR & IIR)**: low/high/band pass, band reject, DC removal, equal loudness
* **Statistical descriptors**: median, mean, variance, power means, raw and central moments, spread, kurtosis, skewness, flatness
* **Time-domain descriptors**: duration, loudness, LARM, Leq, Vickers' loudness, zero-crossing-rate, log attack time and other signal envelope descriptors
* **Spectral descriptors**: Bark/Mel/ERB bands, MFCC, GFCC, LPC, spectral peaks, complexity, roll-off, contrast, HFC, inharmonicity and dissonance
* **Tonal descriptors**: Pitch salience function, predominant melody and pitch, HPCP (chroma) related features, chords, key and scale, tuning frequency
* **Rhythm descriptors**: beat detection, BPM, onset detection, rhythm transform, beat loudness
* **Other high-level descriptors**: danceability, dynamic complexity, audio segmentation, SVM classifier, TensorFlow wrapper for inference
* **Machine learning models**: inference with SVM classifiers and TensorFlow models

You can install Essentia Python from PyPi:

.. code-block::

    pip install essentia

To use TensorFlow models, install via the command:

.. code-block::
   
    pip install essentia-tensorflow


For more details on how to use the library see the documentation section.


Crediting Essentia
==================

Please, credit your use of Essentia properly! If you use the Essentia library in your software, please acknowledge it and specify its origin as <http://essentia.upf.edu>. If you do some research and publish an article, cite both the Essentia paper [1] and the specific references mentioned in the documentation of the algorithms used. We would also be very grateful if you let us know how you use Essentia by sending an email to <mtg@upf.edu>.

[1] Bogdanov, D., Wack N., Gómez E., Gulati S., Herrera P., Mayor O., et al. (2013). `ESSENTIA: an Audio Analysis Library for Music Information Retrieval. <http://hdl.handle.net/10230/32252>`__ International Society for Music Information Retrieval Conference (ISMIR'13). 493-498.


Contributing to Essentia
=========================

We are more than happy to collaborate and receive your contributions to Essentia.
Please see :doc:`contribute` for guidelines.

- `Issue Tracker <https://github.com/MTG/essentia/issues>`_
- `Source Code <https://github.com/MTG/essentia>`_


.. toctree::
   :hidden:
   :caption: Getting Started
   :maxdepth: 1

   introduction
   installing
   download
   essentia_python_tutorial
   python_examples
   FAQ

.. toctree::
   :hidden:
   :caption: Documentation
   :maxdepth: 1

   algorithms_overview
   algorithms_reference
   extractors_out_of_box
   streaming_extractor_music
   machine_learning
   models
   demos


.. toctree::
   :hidden:
   :caption: Advanced Information
   :maxdepth: 1

   design_overview
   howto_standard_extractor
   howto_streaming_extractor
   streaming_architecture

.. toctree::
   :hidden:
   :caption: Extending Essentia
   :maxdepth: 1

   extending_essentia
   extending_essentia_streaming
   composite_api
   execution_network_algorithm
   coding_guidelines
   contribute

.. toctree::
   :hidden:
   :caption: Applications and Licensing
   :maxdepth: 1

   licensing_information
   research_papers
   contents

 