.. How to use TensorFlow models and Gaia SVM classifiers 

Machine learning models
=======================

Essentia includes algorithms for running inference with two types of data-driven machine learning models that can be used for high-level annotation of music audio:

* TensorFlow models
* Gaia's SVM classifiers

We provide various pre-trained models of both types for various music analysis and classification tasks.



Using pre-trained TensorFlow models
-----------------------------------

Essentia provides wrapper algorithms for TensorFlow deep learning models, designed to offer the flexibility of use, easy extensibility, and real-time inference. It allows using virtually any TensorFlow model within our audio analysis framework.

We provide many pre-trained TensorFlow models for auto-tagging, music classification, tempo estimation, source separation, and feature embedding extraction for music and audio in general. See our blog posts `[1] <https://mtg.github.io/essentia-labs/news/tensorflow/2019/10/19/tensorflow-models-in-essentia/>`_ `[2] <https://mtg.github.io/essentia-labs/news/tensorflow/2020/01/16/tensorflow-models-released/>`_ for further details about some of the models.

Our current models include:

* Music auto-tagging (various architectures, trained on the Million Song Dataset and MagnaTagATune datasets).
* Deep embeddings (OpenL3, VGGish-AudioSet).
* Source separation (Spleeter)
* Tempo (BPM) estimation (TempoCNN)
* Transfer learning classifiers

  - music genre (trained on 4 different datasets)
  - moods: happy, sad, aggressive, relaxed, acoustic, electronic, party
  - tonal / atonal
  - danceability
  - voice / instrumental
  - gender (male, female singer)


Installation
^^^^^^^^^^^^

Follow `these instructions <https://mtg.github.io/essentia-labs/news/tensorflow/2019/10/19/tensorflow-models-in-essentia/>`_ to install and use Essentia with the TensorFlow wrapper.

Model downloads
^^^^^^^^^^^^^^^

https://essentia.upf.edu/models/

All the models created by the MTG are available under `the CC BY-NC-ND 4.0 license <https://creativecommons.org/licenses/by-nc-nd/4.0/>`_ and are also available under a proprietary license `upon request <https://www.upf.edu/web/mtg/contact>`_. 


Code examples and demos
^^^^^^^^^^^^^^^^^^^^^^^


See `Python examples of using TensorFlow models <essentia_python_examples.html#inference-with-tensorflow-models>`_.


Some of our models can work in real-time, opening many possibilities for audio developers. For example, `see the demo and code <https://mtg.github.io/essentia-labs/news/tensorflow/2020/04/23/tensorflow-real-time/>`_ for the MusiCNN model performing music auto-tagging on a live audio stream.

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/xMUcY7_n4kQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

    <iframe width="560" height="315" src="https://www.youtube.com/embed/yssBE6oafLs" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Also, see the demos of some of the models `here <demos.html>`_.


Using pre-trained SVM Gaia models
----------------------------------------
Essentia has a wrapper algorithm for LIBSVM for fast inference with SVM models.

We provide various pre-trained SVM classifier models for genres, moods, and instrumentation:

* music genre (trained on 4 different databases)
* ballroom music classification
* moods: happy, sad, aggressive, relaxed, acoustic, electronic, party
* western / non-western music
* tonal / atonal
* danceability
* voice / instrumental
* gender (male, female singer)
* timbre (dark, bright)

These models were trained on annotated music collections, including various inhouse collections created at Music Technology Group. See `more details <https://acousticbrainz.org/datasets/accuracy>`_ regarding their accuracies and the size of the employed datasets for training. To run the models, use the standalone `Music Extractor <streaming_extractor_music.html#high-level-classifier-models>`_. The models are dependent on the version of Essentia, and we currently provide models for both `v2.1_beta1` (compatible with `v2.1_beta2`) and the latest `v2.1_beta5`.

Note that the more recent TensorFlow models now supersede many of the pre-trained SVM models we provide in accuracy.


Installation
^^^^^^^^^^^^
To use the SVM models you need to:

* Install `Gaia2 library <https://github.com/MTG/gaia/blob/master/README.md>`_ (supported on Linux/OSX).
* Build Essentia with examples and Gaia (``--with-examples --with-gaia``).
* Use ``essentia_streaming_extractor_music`` and configure it to include classifier models (see `the detailed documentation <streaming_extractor_music.html>`_).


SVM model downloads
^^^^^^^^^^^^^^^^^^^

https://essentia.upf.edu/svm_models/

All the models created by the MTG are available under `the CC BY-NC-ND 4.0 license <https://creativecommons.org/licenses/by-nc-nd/4.0/>`_ and are also available under a proprietary license `upon request <https://www.upf.edu/web/mtg/contact>`_.


Demos
^^^^^
* `AcousticBrainz <https://acousticbrainz.org>`_ is using our pre-trained SVM classifiers for large-scale music analysis on millions of tracks.

* `AcousticBrainz Moods Playlist Generator <http://mtg.upf.edu/demos/acousticbrainz/moods>`_  is using SVM mood classifiers.


Training your own SVM classifier models in Gaia
-----------------------------------------------

You can train your own SVM classifier models as described below.

To run SVM classification in Essentia you need to prepare a classifier model in Gaia and run the ``GaiaTransform`` algorithm configured to use this model. The example of using high-level models can be seen in the code of ``streaming_music_extractor``. Here we discuss the steps to be followed to train classifier models that can be used with this extractor.

1. Compute music descriptors using ``streaming_music_extractor`` for all audio files.
2. Install Gaia with python bindings.
3. Prepare JSON `groundtruth <https://github.com/MTG/gaia/blob/master/src/bindings/pygaia/scripts/classification/groundtruth_example.yaml>`_ and `filelist <https://github.com/MTG/gaia/blob/master/src/bindings/pygaia/scripts/classification/filelist_example.yaml>`_ files (see examples).
    - Groundtruth file maps identifiers for audio files (they can be paths to audio files or whatever id strings you want to use) to class labels. 
    - Filelist file maps these identifiers to the actual paths to the descriptor files for each audio track. 
4. Currently, Gaia does not support loading descriptors in JSON format. As a workaround, you can configure the extractor output to YAML format in Step 1, or run ``json_to_sig.py`` `conversion script <https://github.com/MTG/gaia/blob/master/src/bindings/pygaia/scripts/classification/json_to_sig.py>`_.
5. Run ``train_model.py`` script in Gaia (`here <https://github.com/MTG/gaia/blob/master/src/bindings/pygaia/scripts/classification/train_model.py>`_) with these groundtruth and filelist files. The script will create the classifier model file. 

6. The model file can now be used by a GaiaTransform algorithm inside ``streaming_music_extractor``. 

Alternatively to steps 3-5, you can use a simplified `script <https://github.com/MTG/gaia/blob/master/src/bindings/pygaia/scripts/classification/train_model_from_sigs.py>`_ that trains a model given a folder with sub-folders corresponding to class names and containing descriptor files for these classes. 

Note that using a specific classifier model implies that you are expected to give a pool with the same descriptor layout as the one used in training as an input to the ``GaiaTransform`` algorithm.

How it works
^^^^^^^^^^^^

To train the SVMs Gaia internally uses the `LibSVM <https://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_ library. The training script automatically creates an SVM model given a ground-truth dataset using the best combination of parameters for data preprocessing and SVM that it can find in a grid search. Testing all possible combinations the script conducts 5-fold cross-validation for each one of them: The ground-truth dataset is randomly split into train and test sets, the model is trained on the train set and is evaluated on the test set. Results are averaged across 5 folds including the confusion matrix. After all combinations of parameters have been evaluated, the winning combination is selected according to the best accuracy obtained in cross-validation and the final SVM classifier model is trained using *all* ground-truth data. See the "Cross-validation and Grid-search" section in the `practical guide to SVM classification <https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf>`_ for more details.

The combinations of parameters tested in a grid search by default are mentioned `in the code <https://github.com/MTG/gaia/blob/master/src/bindings/pygaia/scripts/classification/classification_project_template.yaml>`_. Users can modify these parameters according to their needs by creating such a classification project file on their own.

The parameters include:

- SVM kernel type: polynomial or RBF
- SVM type: currently only C-SVC
- SVM C and gamma parameters
- preprocessing type:

- use all descriptors, no preprocessing
- use ``lowlevel.*`` descriptors only
- discard energy bands descriptors (``*barkbands*``, ``*energyband*``, ``*melbands*``, ``*erbbands*``)
- use all descriptors, normalize values
- use all descriptors, normalize and gaussianize values

- number of folds in cross-validation: 5 by default

In the preprocessing stage, the training script loads all descriptor files according to the preprocessing type. Additionally, some descriptors are always ignored, including all ``metadata*`` that is the information not directly associated with audio analysis. The ``*.dmean``, ``*.dvar``, ``*.min``, ``*.max``, ``*.cov`` descriptors are also ignored, and therefore, currently only means and variances are used for descriptors summarized across frames. Non-numerical descriptors are enumerated (``tonal.chords_key``, ``tonal.chords_scale``, ``tonal.key_key``, ``tonal.key_scale``).

Note that cross-validation script splits the ground-truth dataset into train and test sets randomly. In the case of music classification tasks, one may want to assure artist/album filtering (that is, no artist/album occurs in the test set if it occurs in train set). The current way to achieve it is to ensure that the whole input dataset contains only one item per artist/album. Alternatively, you can adapt the scripts to suit your needs.

How to train an SVM model with a different set of parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our training script generates a single model retrained on the whole dataset with the best parameters combination from the grid search. However, you may want to generate new models with custom parametrizations. Imagine, for instance, that you need a model that runs on a lighter set of features despite the accuracy drop, or that you believe that a different parameter set can improve results for your particular scenario.

To generate a model given the ``<project_file>`` and your chosen ``<param_file>`` from the results folder, execute the following python lines::

  from gaia2.scripts.classification.retrain_model import retrainModel
  retrainModel(project_file, param_file, output_file)

This creates a Gaia model and saves it into ``<output_file>``. 

Also, note that the ``retrain_model`` can be called as a command-line program.


How to choose a parameter configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the end of the training process, a file called ``<project_name>.report.csv`` is created. It provides a ranking in terms of accuracy and normalized accuracy as well as the standard deviation between folds for every set of parameters. By having a look at this file you can get some insights about which parameters to try. You can, for instance, estimate the expected accuracy drop if you decide to go for a configuration with a smaller set of descriptors.
