.. highlight:: cpp

How to write a simple extractor using the streaming mode of Essentia
====================================================================

The goal of this howto/tutorial is to show you how to write extractors in the streaming
mode of Essentia. To this end, we will write an extractor that extracts the MFCCs of
an audio file, computes their average, variance, min and max, and outputs that to a file.

This tutorial is the equivalent, and intend to achieve the same goal, as the
:doc:`howto_standard_extractor` tutorial.

**Note:** the source code for this example can be found in the git repository tree, in the
:download:`src/examples/streaming_mfcc.cpp <../../src/examples/streaming_mfcc.cpp>` file.

First of all, let's identify which algorithms we will need. We want to do the following processing:

.. image:: _static/mfcc_extractor_halfsize.png

The steps we will have to take are the following:

* instantiate these Algorithms (from the ``streaming::AlgorithmFactory``)
* (possibly) configure them
* connect them through their sinks/sources, make sure no sink/source is left unconnected
* create a ``Network`` of those algorithms, and launch the whole processing with the ``run()`` method

You will notice that compared to the standard mode, here we don't have to do anything in
particular once everything is connected. The order of processing is automatically decided
by the scheduler, and hence, this mode of operation looks a lot like `functional`_ programming.


Setting up our program
----------------------

Let's start again by examining the source code for the streaming_mfcc.cpp example::

  using namespace essentia::streaming;

  int main(int argc, char* argv[]) {

    if (argc != 3) {
      cout << "ERROR: incorrect number of arguments." << endl;
      cout << "Usage: " << argv[0] << " audio_input yaml_output" << endl;
      exit(1);
    }

    string audioFilename = argv[1];
    string outputFilename = argv[2];

    // register the algorithms in the factory(ies)
    essentia::init();

    Pool pool;

    /////// PARAMS //////////////
    Real sampleRate = 44100.0;
    int frameSize = 2048;
    int hopSize = 1024;


This is the same boilerplate code as in the standard example, so we won't spend too much time on it.
The namespace is ``essentia::streaming`` here, though.


Creating the required algorithms
--------------------------------

::

  // we want to compute the MFCC of a file: we need the create the following:
  // audioloader -> framecutter -> windowing -> FFT -> MFCC -> PoolStorage

  AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate,
                                    "silentFrames", "noise");

  Algorithm* fc    = factory.create("FrameCutter",
                                    "frameSize", frameSize,
                                    "hopSize", hopSize);

  Algorithm* w     = factory.create("Windowing",
                                    "type", "blackmanharris62");

  Algorithm* spec  = factory.create("Spectrum");
  Algorithm* mfcc  = factory.create("MFCC");


This is also very similar to the standard example, however note the differences:

* the factory is now the ``essentia::streaming::AlgorithmFactory`` instead of the ``essentia::standard::AlgorithmFactory``
* the algorithm type is now ``essentia::streaming::Algorithm`` instead of ``essentia::standard::Algorithm``


Connecting the algorithms
-------------------------

::

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  // Audio -> FrameCutter
  audio->output("audio")    >>  fc->input("signal");

  // FrameCutter -> Windowing -> Spectrum
  fc->output("frame")       >>  w->input("frame");
  w->output("frame")        >>  spec->input("frame");

  // Spectrum -> MFCC -> Pool
  spec->output("spectrum")  >>  mfcc->input("spectrum");

  mfcc->output("bands")     >>  NOWHERE;                   // we don't want the mel bands
  mfcc->output("mfcc")      >>  PC(pool, "lowlevel.mfcc"); // store only the mfcc coeffs

  // Note: PC is a #define for PoolConnector


Here goes the connection of the algorithms. In streaming mode, you do not need an intermediate
variable to connect the output of an algorithm and the input of another one on it, you simply
connect the output of an algorithm directly to its corresponding input. You can either use
the ``connect(input, output)`` function or the ``>>`` right-shift operator to connect an
input to an output. In this example, we use the ``>>`` operator, because it looks nicer!

Note the special connector ``NOWHERE``, which you need to specify. It is mandatory to connect
all inputs/outputs, so if you want to discard one stream, you need to explicitly say it by
connecting it to the ``NOWHERE`` connector. Failure to do so will result in an exception when
you try to run the network.

You can also see another special connector on the next line, that allows you to store the
output of an algorithm in a ``Pool``, where you then specify the pool and descriptor
name, and it will automatically get stored there as soon as it becomes available on the
given output.


Processing the audio
--------------------

::

  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  // create a network with our algorithms...
  Network n(audio);
  // ...and run it, easy as that!
  n.run();


You need to create a network of algorithms by constructing it with the topmost algorithm
in your processing tree, that is the audio loader (all algorithms are connected after it).
The audio loader is referred to as the *generator* in this case.

And this is all you have to call to make all the processing happen. Basically, all the algorithms
will do all the processing they can (that is, compute all the MFCCs for all the audio), and
when the ``run()`` function returns, the Pool will be filled with the MFCC coefficients.


Aggregating the results and writing them to disk
------------------------------------------------

::

  // aggregate the results
  Pool aggrPool; // the pool with the aggregated MFCC values
  const char* stats[] = { "mean", "var", "min", "max" };

  standard::Algorithm* aggr = standard::AlgorithmFactory::create("PoolAggregator",
                                                                 "defaultStats", arrayToVector<string>(stats));

  aggr->input("input").set(pool);
  aggr->output("output").set(aggrPool);
  aggr->compute();

  // write results to file
  cout << "-------- writing results to file " << outputFilename << " --------" << endl;

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", outputFilename);
  output->input("pool").set(aggrPool);
  output->compute();


At this point, the processing is the same as in the standard example: aggregate
and output data to file. Note that the ``PoolAggregator`` and the ``YamlOutput``
still come from the non-streaming (standard) factory. ::

  n.clear();
  delete aggr;
  delete output;
  essentia::shutdown();

  return 0;


And the cleanup part, which is also quite simplified with respect to the way it's done in
the standard way. As all the algorithms are connected in a network, you just need to call
the :essentia:`Network::clear()` method to delete all of them.

You also need to delete the ``PoolAggregator`` and ``YamlOutput`` which you allocated separately,
call :essentia:`shutdown()`, and you're done!


.. _functional: http://en.wikipedia.org/wiki/Functional_programming


Compiling extractor
-------------------

Follow the same instructions as for `standard extractors <howto_standard_extractor.html#compiling-extractors>`_ 
in order to compile your extractor. 
