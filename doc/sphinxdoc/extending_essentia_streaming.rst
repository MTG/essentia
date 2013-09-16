.. highlight:: cpp

How to write a StreamingAlgorithm
=================================

There are many ways to write a streaming algorithm, all of which involve deriving from
the base class ``StreamingAlgorithm`` or one of its subclasses.
Some methods are easier than others, some are more powerful. They are listed here in
order of simplicity.



Derive from StreamingAlgorithmWrapper
-------------------------------------

This method is by far the easiest, and will actually allow you to wrap already existing
algorithms in 90% of the cases. As a prerequisite, you will need to have a "standard"
algorithm working. This way of doing is especially recommended for frame-based algorithms
(ie: FFT, Centroid, PCP, etc...), but will also work for audio filters.

To wrap it, you will need to:

* derive from ``StreamingAlgorithmWrapper``
* declare your Sinks and Sources
* declare the name of the Algorithm you're wrapping
* and that's it!

All the parameters will be forwarded to the wrapped class, and basically the new streaming
algorithm will work exactly as the old one, except that you can now use it in a
streaming environment!

As an example, let's look at the spectral centroid::

  class Centroid : public StreamingAlgorithmWrapper {

   protected:
    Sink<std::vector<Real> > _array;
    Source<Real> _centroid;

   public:
    Centroid() {
      declareInput(_array, TOKEN, "array");
      declareOutput(_centroid, TOKEN, "centroid");
      declareAlgorithm("Centroid");
    }
  };


The first argument of ``declareInput()`` / ``declareOutput()`` is obviously the Sink/Source object.

The second argument needs a bit more explanation: it tells the wrapper whether the
``Algorithm`` you are wrapping was taking only one token from the input stream, or many of them.

In this case, TOKEN indicates that the wrapper will call the centroid algorithm, passing
it *1* single token as input. Let's have a quick look at the "standard" Centroid code::

  class Centroid : public Algorithm {

   protected:
    Input<std::vector<Real> > _array;
    Output<Real> _centroid;

   public:
    Centroid() {
      declareInput(_array, "array", "the input array");
      declareOutput(_centroid, "centroid", "the centroid of the array");
    }

    void declareParameters() {
      declareParameter("range", "the range of the input array, used for normalizing the results", 1.0);
    }
  };


we can see that the Input and the Output is of the exact same type as the Sink and Source
of the streaming version. Hence the ``TOKEN`` argument, which actually corresponds to doing
something like this (this is not working code, it is pseudo code/C++ for understanding)::

  vector<Real> array = _array.getOneFrameForReading();
  Real centroid;

  _wrappedCentroidAlgo->input("array").set(array);
  _wrappedCentroidAlgo->output("centroid").set(centroid);
  _wrappedCentroidAlgo->compute();

  _centroid.produceOneFrame(centroid);



The other value that the second argument can take is ``STREAM``, which means that the
wrapped algorithm was already working on a stream of tokens. In that case, you also need
to specify how many tokens to wait for before calling the wrapped algorithm. An example
will be more telling, so let's have a look at the Scale algorithm code (again, simplified)::


  class Scale : public Algorithm {

   protected:
    Input<std::vector<Real> > _signal;
    Output<std::vector<Real> > _scaled;

   public:
    Scale() {
      declareInput(_signal, "signal", "the input signal");
      declareOutput(_scaled, "signal", "the scaled signal");
    }
  };

  namespace streaming {

  class Scale : public StreamingAlgorithmWrapper {

   protected:
    Sink<Real> _signal;
    Source<Real> _scaled;

   public:
    Scale() {
      int preferredSize = 4096;
      declareInput(_signal, preferredSize, STREAM, "signal");
      declareOutput(_scaled, preferredSize, STREAM, "signal");
      declareAlgorithm("Scale");
    }
  };

  } // namespace streaming


Here, what happens is slightly more complex (but not so much!). The "standard" algorithm
expects a ``vector<Real>`` as argument, but the streaming algorithm takes a flow of ``Real``.
So, why is it different now? This happens because in the standard way, we're not working on
single tokens anymore, but a bunch of them which have already been put into a vector,
most probably for performance reasons.

The ``StreamingAlgorithmWrapper`` can do the same operation for us automatically, but we
need to tell it to do so, and we also need to give it a predefined size so that the scheduler
knows how many tokens to wait for before calling the algorithm.

This is done by specifying ``4096, STREAM``, instead of ``TOKEN`` in the declareInput/Output
functions. This means that the Scale algorithm will be called on buffers of size 4096,
as soon as that many tokens are available on the input Sink.



Derive from StreamingAlgorithmComposite
---------------------------------------

Deriving from ``StreamingAlgorithmComposite`` allows you to create blocks of algorithms,
which is nice to encapsulate functionality while still keeping the modularity of small
algorithms.
You can thus wrap a long and complex network of algorithms which does some very complex
task into a single black-box, which can later be used as a single algorithm while keeping
the advantage of the streaming mode (everything stays multi-threaded, etc...)

Please take a look at the code of the MonoLoader algorithm as an example.

The MonoLoader actually does the following: AudioLoader -> MonoMixer -> Resample.

Internally, what the we do is we connect these 3 algorithms as if it was an extractor, and
declare which are the inputs/outputs which need to be visible.

You do this as usual with the ``declareInput`` and ``declareOutput`` method, passing it an
already existing connector and giving it a new name::

  declareOutput(_innerAlgo->output("signal"), "signal");


This tells that the output of the inner algorithm which is called "signal" should be an
externally visible output for the composite algorithm, with the name "signal" also
(could have been a different one).

From an outside point of view, this just looks like a single ``StreamingAlgorithm``, when
in fact it is a "subnetwork" of processing.


Declaring your generators
^^^^^^^^^^^^^^^^^^^^^^^^^

There is one important thing to know when writing composite algorithms, and that is
necessary only when you have generators inside of your composite algorithm:
(Such is the case for the MonoLoader, because the AudioLoader is a generator).

*You have to declare your generators by putting them in the member variable ``StreamingAlgorithmComposite::_generators``.*

If you forget to do that, the scheduler will be unable to work correctly. It is not necessary
to do this for any other algorithm, because they are all connected. Generators are the only
algorithms that do not have a "parent", and so they need to be treated separately.



Derive from StreamingAlgorithm
------------------------------

This is the most barebones way to define a ``StreamingAlgorithm``, and as such the most
difficult to master, but also the most powerful.
It requires you to grasp a few more concepts of what is going on in Essentia, mainly how
the consumption model works and how algorithms get scheduled.

Please refer to the `streaming architecture <streaming_architecture>`_ for an
explanation of these concepts.


More about the consumption model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before actually processing the data, you need to acquire it. You saw in the
:doc:`design overview </streaming_architecture>` page that ``Sinks`` and ``Sources`` do this
by calling the ``StreamConnector::acquire(int n_tokens)`` method.

For convenience, you can define a current acquire size (and release size) for each
``Source`` and ``Sink``, so that it is possible to just call ``StreamConnector::acquire()``
without arguments, all of which calls can then be factored into one single invocation of
``StreamingAlgorithm::acquireData()``.

This method will return any of these 3 values, which are part of the enum ``SyncStatus``:

* ``SYNC_OK``, meaning that you could acquire the required number of tokens on all
  ``Sinks`` and ``Sources``
* ``NO_INPUT``, meaning that there was at least one ``Sink`` for which you could not
  acquire the required number of tokens. In general, this means that you processed all
  the input data that you could, and that you should just simply return from the function
* ``NO_OUTPUT``, meaning that there was at least one ``Source`` for which you could not
  acquire the required number of tokens. In general, this means that the output buffer is
  full, so you should either use a bigger buffer, or it can mean that you have a problem
  in your scheduling (producing too much, or a connected algorithm that don't consume
  correctly what your algorithm is producing)

The equivalent method to release everything when you're done with it is the
``StreamingAlgorithm::releaseData()`` method.


StreamingAlgorithm behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The expected behavior of a ``StreamingAlgorithm`` to be correctly scheduled is the following:

**Whenever the process() method gets called, the algorithm should process as much as possible
of the data that is available on its sinks and return ``true`` if it produced some data,
or ``false`` if it didn't.**

Your streaming algorithms should **always** conform to this behavior.

Hence it is highly recommended to have a ``process()`` method that looks like the following one::


  bool Algo::process() {
    bool producedData = false;

    while (true) {
      SyncStatus status = acquireData();
      if (status != SYNC_OK) {
        // acquireData() returns SYNC_OK if we could reserve both inputs and outputs
        // being here means that there is either not enough input to process,
        // or that the output buffer is full, in which cases we need to return from here
        return producedData;
      }

      // do stuff here
      ...

      // give back the tokens that were reserved
      releaseData()

      producedData = true;
    }
  }



Examples
^^^^^^^^

The theory is all there, but it will probably still look very abstract to you. The best way
to explain further is probably to show examples, so here is a list of algorithms which derive
directly from ``StreamingAlgorithm``, with the complexity of their implementation indicated
inside parentheses:

- Monomixer *(easy)*
  (:download:`monomixer.h <../../src/algorithms/standard/monomixer.h>` and
  :download:`monomixer.cpp <../../src/algorithms/standard/monomixer.cpp>`)
- Resample *(medium)*
  (:download:`resample.h <../../src/algorithms/standard/resample.h>` and
  :download:`resample.cpp <../../src/algorithms/standard/resample.cpp>`)
- Trimmer *(medium)*
  (:download:`trimmer.h <../../src/algorithms/standard/trimmer.h>` and
  :download:`trimmer.cpp <../../src/algorithms/standard/trimmer.cpp>`)
- Slicer *(hard)*
  (:download:`slicer.h <../../src/algorithms/standard/slicer.h>` and
  :download:`slicer.cpp <../../src/algorithms/standard/slicer.cpp>`)
- FrameCutter *(insanely hard)*
  (:download:`framecutter.h <../../src/algorithms/standard/framecutter.h>` and
  :download:`framecutter.cpp <../../src/algorithms/standard/framecutter.cpp>`)
