.. How to use TensorFlow models and Gaia SVM classifiers 

Using machine learning models
==============================

Essentia includes algorithms for running inference with data-driven machine learning models that can be used for high-level annotation of music audio.
Specifically, Essentia provides a wrapper for TensorFlow that allows using virtually any TensorFlow model within our audio analysis framework.

We provide pre-trained models for various music analysis and classification tasks.
Current :ref:`Essentia Models` are based on TensorFlow.

*Note*: We also provide legacy `Gaia SVM models <gaia_svm_models.html>`_ based on handcrafted music audio features.
These models have been superseded by our current models.


Installation
------------

TensorFlow inference support is available for Linux (x86_64, aarch64) and macOS
(x86_64, arm64) as a separate Python package, `essentia-tensorflow
<https://pypi.org/project/essentia-tensorflow/>`_:

.. highlight:: none

.. code-block::

    pip install essentia-tensorflow

``essentia-tensorflow`` is a superset of ``essentia``: install **exactly one of the two**,
never both, since both install into the same ``essentia`` Python package name and the one
installed second shadows the first. There is no separate ``essentia.tensorflow`` module,
and there never has been one -- the TensorFlow algorithms simply appear alongside every
other algorithm in ``essentia.standard`` and ``essentia.streaming`` once
``essentia-tensorflow`` is the package installed:

.. code-block::

    from essentia.standard import TensorflowPredictEffnetDiscogs

See `Installing from PyPI (wheels) <installing.html#installing-from-pypi-wheels>`_ for the
full supported-platform matrix, including which TensorFlow C library each platform's wheel
vendors.

Building Essentia with TensorFlow support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build Essentia from source with TensorFlow support instead, see `Installing TensorFlow
<installing.html#installing-tensorflow>`_ in the installation guide: it covers providing a
TensorFlow C library (a packaged ``libtensorflow``, a pip TensorFlow wheel, or
``packaging/fetch_libtensorflow.sh``, the reproducible path the published wheels use), the
``--with-tensorflow`` configure flag, and the configure-time link test that catches a
``tensorflow.pc`` naming the wrong library before the build starts.


Inference with GPU
--------------------
It is possible to run inference with Essentia Models using GPU when the correct version of the CUDA and CuDNN libraries are installed on your system.
We recommend using a package manager such as `Conda <https://docs.conda.io/en/latest/>`_ to install the required components.

These are the recommended steps to follow:

Install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ (or Anaconda).
Create and activate a Conda environment: 

.. code-block::

    conda create -n ess python=3.10
    conda activate ess

Install CUDA, CuDNN, and essentia-tensorflow:

.. code-block::

    conda install -c conda-forge -y cudatoolkit=11.2 cudnn=8.1
    pip install essentia-tensorflow


It is possible to use CUDA environment variables to control GPU usage.
For example, the following line of code launches a script using GPU 1:

.. code-block::

  CUDA_VISIBLE_DEVICES=1 python my_script.py 

.. highlight:: default
