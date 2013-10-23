.. highlight:: python

Python beginner tutorial
========================

Introduction
------------

This is a hands-on tutorial for complete newcomers to Essentia. The source code for
this tutorial can be found in the ``examples/tutorial`` folder, in the
:download:`essentia_tutorial.py <../../src/examples/tutorial/essentia_tutorial.py>` file.

Essentia is written in C++ but it has python bindings that makes it very well suited
for use by scientists which are used to Matlab, for instance.

You can run all of the following using the standard python interpreter, however
it is highly recommended to use a better one, such as `IPython <http://ipython.org/>`_
*(recommended)* or `bpython <http://bpython-interpreter.org/>`_. Some users might also 
like `Spyder <http://code.google.com/p/spyderlib/>`_ environment, as it mimics Matlab 
(and looks even better, although the authors have no experience with it at the moment).

You should have the `NumPy <http://numpy.scipy.org/>`_ package installed, which gives
Python the ability to work with vectors and matrices in much the same way as Matlab. You 
can also install `SciPy <http://www.scipy.org/>`_, which provides functionality similar 
to Matlab's toolboxes, although we won't be using it in this tutorial.

You should have the `matplotlib <http://matplotlib.sourceforge.net/>`_ package
installed if you want to be able to do some plotting.

**Note:** this tutorial demonstrates the standard mode of Essentia (think Matlab).
There is another tutorial for the streaming mode, which you can find in the
:download:`essentia_tutorial_streaming.py <../../src/examples/tutorial/essentia_tutorial_streaming.py>`
file, but which is not covered in this documentation.


Exploring the python module
---------------------------

**Note:** all the following commands need to be typed in a python interpreter. It is highly
recommended to use IPython, and to start it with the ``--pylab`` option to have
interactive plots.

In this tutorial, we will have a look at some basic functionality:
 - how to load an audio
 - how to perform some numerical operations, such as FFT, ...
 - how to plot results
 - how to output results to a file

But first, let's investigate a bit the Essentia package::

  # first, we need to import our essentia module. It is aptly named 'essentia'!

  import essentia

  # as there are 2 operating modes in essentia which have the same algorithms,
  # these latter are dispatched into 2 submodules:

  import essentia.standard
  import essentia.streaming

  # let's have a look at what is in there

  print dir(essentia.standard)

  # you can also do it by using autocompletion in IPython, typing "essentia.standard." and pressing Tab

Instantiating our first algorithm, loading some audio
-----------------------------------------------------

Let's start doing some useful things now!

Before you can use algorithms in Essentia, you first need to instantiate (create) them.
When doing so, you can give them parameters which they may need to work properly,
such as the filename of the audio file in the case of an audio loader.

Once you have instantiated an algorithm, nothing has happened yet, but your algorithm
is ready to be used and works like a function, that is, *you have to call it to make
stuff happens* (technically, it is a `function object <http://en.wikipedia.org/wiki/Function_object>`_).

Which gives::

  # Essentia has a selection of audio loaders:
  #
  #  - AudioLoader: the basic one, returns the audio samples, sampling rate and number of channels
  #  - MonoLoader: which returns audio, down-mixed and resampled to a given sampling rate
  #  - EasyLoader: a MonoLoader which can optionally trim start/end slices and rescale according
  #                to a ReplayGain value
  #  - EqloudLoader: an EasyLoader that applies an equal-loudness filtering on the audio
  #

  # we start by instantiating the audio loader:
  loader = essentia.standard.MonoLoader(filename = 'audio.mp3')

  # and then we actually perform the loading:
  audio = loader()

  # by default, the MonoLoader will output audio with 44100Hz samplerate

and to make sure that this actually worked, let's plot a 1-second slice of audio, from
t = 1sec to t = 2sec::

  # pylab contains the plot() function, as well as figure, etc... (same names as Matlab)
  from pylab import plot, show, figure

  plot(audio[1*44100:2*44100])
  show() # unnecessary if you started "ipython --pylab"


Note how the indexing syntax in python is the same as in Matlab, the only difference being that
indices start at 0 (similar to nearly all the programming languages), while they start
at 1 in Matlab.

Also note that if you have started IPython with the ``--pylab`` option, the call to
show() is not necessary, and you don't have to close the plot to regain control of your terminal.


Setting the stage for our future computations
---------------------------------------------

So let's say that we want to compute the `MFCCs <http://en.wikipedia.org/wiki/Mel-frequency_cepstral_coefficient>`_
for the frames in our audio.

We will need the following algorithms: Windowing, Spectrum, MFCC ::

  from essentia.standard import *
  w = Windowing(type = 'hann')
  spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
  mfcc = MFCC()

  # let's have a look at the inline help:
  help(MFCC)

  # you can also see it by typing "MFCC?" in IPython

And remember that once algorithms have been instantiated, they work like normal functions::

  frame = audio[5*44100 : 5*44100 + 1024]
  spec = spectrum(w(frame))

  plot(spec)
  show() # unnecessary if you started "ipython --pylab"



Computing MFCCs the Matlab way
------------------------------

Now let's compute the MFCCs the way we would do it in Matlab, slicing the frames manually::

  mfccs = []
  frameSize = 1024
  hopSize = 512

  for fstart in range(0, len(audio)-frameSize, hopSize):
      frame = audio[fstart:fstart+frameSize]
      mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
      mfccs.append(mfcc_coeffs)

  # and plot them...
  # as this is a 2D array, we need to use imshow() instead of plot()
  imshow(mfccs, aspect = 'auto')
  show() # unnecessary if you started "ipython --pylab"

See also that the MFCC algorithm returns 2 values: the band energies and the coefficients, and
that you get (unpack) them the same way as in Matlab.

Let's see if we can write this in a nicer way, though.


Computing MFCCs the Essentia way
--------------------------------

Essentia has been designed to do audio processing, and as such it has lots of readily available 
related algorithms ; you don't have to chase around lots of toolboxes to be able to achieve what you want.

For more details, it is recommended to have a look either at the :doc:`algorithms_overview`
or at the `complete reference`_.

In particular, we will use the ``FrameGenerator`` here::

  mfccs = []

  for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
      mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
      mfccs.append(mfcc_coeffs)

  # transpose to have it in a better shape
  # we need to convert the list to an essentia.array first (== numpy.array of floats)
  mfccs = essentia.array(mfccs).T

  # and plot
  imshow(mfccs[1:,:], aspect = 'auto') 
  show() # unnecessary if you started "ipython --pylab"

  # We ignored the first MFCC coefficient to disregard the power of the signal and only plot its spectral shape


Introducing the Pool - a versatile data container
-------------------------------------------------

A ``Pool`` is a container similar to a C++ map or Python dict which can contain any
type of values (easy in Python, not as much in C++...). Values are stored in there
using a name which represent the full path to these values; dot ('.') characters are
used as separators. You can think of it as a directory tree, or as namespace(s) + local name.

Examples of valid names are: ``"bpm"``, ``"lowlevel.mfcc"``, ``"highlevel.genre.rock.probability"``, etc...

So let's redo the previous computations using a pool::

  pool = essentia.Pool()

  for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
      mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
      pool.add('lowlevel.mfcc', mfcc_coeffs)
      pool.add('lowlevel.mfcc_bands', mfcc_bands)

  imshow(pool['lowlevel.mfcc'].T[1:,:], aspect = 'auto')
  show() # unnecessary if you started "ipython --pylab"
  figure()
  imshow(pool['lowlevel.mfcc_bands'].T, aspect = 'auto', interpolation = 'nearest')


The pool also has the nice advantage that the data you get out of it is already in
an ``essentia.array`` format (which is equal to numpy.array of floats), so you can 
call transpose (``.T``) directly on it.


Aggregation and file output
---------------------------

Let's finish this tutorial by writing our results to a file. As we are using such a
nice language as Python, we could use its facilities for writing data to a file, but
for the sake of this tutorial let's do it using the ``YamlOutput`` algorithm,
which writes a pool in a file using the `YAML <http://yaml.org/>`_ or 
`JSON <http://en.wikipedia.org/wiki/JSON>_` format. ::

  output = YamlOutput(filename = 'mfcc.sig') # use "format = 'json'" for JSON output
  output(pool)

  # or as a one-liner:
  YamlOutput(filename = 'mfcc.sig')(pool)

This should take a while as we actually write the MFCCs for all the frames, which
can be quite heavy depending on the duration of your audio file.

Now let's assume we do not want all the frames but only the mean and variance of
those frames. We can do this using the ``PoolAggregator`` algorithm and use it
on the pool to get a new pool with the aggregated descriptors::

  # compute mean and variance of the frames
  aggrPool = PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)

  print 'Original pool descriptor names:'
  print pool.descriptorNames()
  print
  print 'Aggregated pool descriptor names:'
  print aggrPool.descriptorNames()

  # and ouput those results in a file
  YamlOutput(filename = 'mfccaggr.sig')(aggrPool)



And that closes the tutorial session!

There is not much more to know about Essentia for using it in python environment, the basics are:

* instantiate and configure algorithms
* use them to compute some results
* and that's pretty much it!

The big strength of Essentia is that it provides a considerably large collection of algorithms,
from low-level to high-level descriptors, which have been thoroughly optimized and
tested and which you can rely on to build your own signal analysis.

The following steps which you might want to take are:

* study the :download:`streaming python tutorial <../../src/examples/tutorial/essentia_tutorial_streaming.py>` file
* look at the :doc:`algorithms_overview` or the `complete reference`_.
* check more python examples
    * standard mode:
        * :download:`extractor_spectralcentroid.py <../../src/examples/tutorial/extractor_spectralcentroid.py>`
        * :download:`onsetdetection_example.py <../../src/examples/tutorial/onsetdetection_example.py>`
        * :download:`extractor_predominantmelody.py <../../src/examples/tutorial/extractor_predominantmelody.py>`
        * :download:`extractor_predominantmelody_by_steps.py <../../src/examples/tutorial/extractor_predominantmelody_by_steps.py>`
    * streaming mode:
        * :download:`streaming_extractor_keyextractor_by_steps.py <../../src/examples/tutorial/streaming_extractor_keyextractor_by_steps.py>`
* read the C++ tutorial for :doc:`howto_standard_extractor` or :doc:`howto_streaming_extractor`
* become a developer and write algorithms yourself! (see links on the `first page <index.html>`_, in the developer section)

.. _complete reference: algorithms_reference.html
