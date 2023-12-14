Interactive demos
=================

Music audio descriptors in the browser
--------------------------------------

Examples of music audio analysis with Essentia algorithms using Essentia.js

https://mtg.github.io/essentia.js/examples/


Tempo estimation
----------------

Tempo BPM estimation with Essentia: https://replicate.com/mtg/essentia-bpm


Essentia TensorFlow models
--------------------------

Examples of inference with the pre-trained TensorFlow models for music auto-tagging and classification tasks:

- Music classification by genre, mood, danceability, instrumentation: https://replicate.com/mtg/music-classifiers
- Music style classification with the Discogs taxonomy (400 styles, MAEST model). Overall track-level predictions: https://replicate.com/mtg/maest
- Music style classification with the Discogs taxonomy (400 styles, Effnet-Discogs model). Overall track-level predictions: https://replicate.com/mtg/effnet-discogs
- Music style classification with the Discogs taxonomy (400 styles, Effnet-Discogs model). Segment-level real-time predictions with Essentia.js: https://essentia.upf.edu/essentiajs-discogs
- Real-time music autotagging (50 tags) in the browser with Essentia.js: https://mtg.github.io/essentia.js/examples/demos/autotagging-rt/
- Mood classification in the browser with Essentia.js: https://mtg.github.io/essentia.js/examples/demos/mood-classifiers/
- Music emotion arousal/valence regression: https://replicate.com/mtg/music-arousal-valence
- Music approachability and engagement: https://replicate.com/mtg/music-approachability-engagement
- Jupyter notebook for real-time music `auto-tagging <https://github.com/MTG/essentia/blob/master/src/examples/python/tutorial_tensorflow_real-time_auto-tagging.ipynb>`_ and `classification <https://github.com/MTG/essentia/blob/master/src/examples/python/tutorial_tensorflow_real-time_simultaneous_classifiers.ipynb>`_.

    .. raw:: html

        <iframe width="560" height="315" src="https://www.youtube.com/embed/xMUcY7_n4kQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

        <iframe width="560" height="315" src="https://www.youtube.com/embed/yssBE6oafLs" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Essentia SVM models
-------------------

Examples of inference with older SVM models for music classification tasks:

- `AcousticBrainz <https://acousticbrainz.org>`_ is using our pre-trained SVM classifiers for large-scale music analysis on millions of tracks.
- `AcousticBrainz Moods Playlist Generator <http://mtg.upf.edu/demos/acousticbrainz/moods>`_  is using SVM mood classifiers.
