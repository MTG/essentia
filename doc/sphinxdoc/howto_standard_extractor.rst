.. highlight:: cpp

How to write a simple extractor using the standard mode of Essentia
===================================================================

The goal of this howto/tutorial is to show you how to write extractors in the standard
mode of Essentia. To this end, we will write an extractor that extracts the MFCCs of
an audio file, computes their average, variance, min and max, and outputs that to a file.

If you are more interested in the streaming mode, please have a look at the :doc:`howto_streaming_extractor` tutorial.

**Note:** the source code for this example can be found in the git repository tree,
:download:`src/examples/standard_mfcc.cpp <../../src/examples/standard_mfcc.cpp>` file.

First of all, let's identify which algorithms we will need. We want to do the following processing:

.. image:: _static/mfcc_extractor_halfsize.png

We will have to take the following steps:

* instantiate these Algorithms
* (possibly) configure them
* connect their inputs/outputs to the variables they will use for processing
* call their ``compute()`` method to get the MFCC values for each frame
* store computed values in a ``Pool``
* at the end, output the results of the aggregation of the values in the Pool

As we explicitly tell the computer which action to do at each step, the
standard mode of Essentia is `imperative`_.


Setting up our program
----------------------

So let's start by examining the source code for the standard_mfcc.cpp example::

  using namespace essentia::standard;

  int main(int argc, char* argv[]) {

    if (argc != 3) {
      cout << "ERROR: incorrect number of arguments." << endl;
      cout << "Usage: " << argv[0] << " audio_input yaml_output" << endl;
      exit(1);
    }

    string audioFilename = argv[1];
    string outputFilename = argv[2];



This is some boilerplate code, and you shouldn't have too much trouble understanding
it. It basically gets the input and output filename from the command line. Note though
that we will use the ``essentia::standard`` namespace, which contains the ``Algorithm``
class as well as the ``AlgorithmFactory``. ::

    // register the algorithms in the factory(ies)
    essentia::init();

    Pool pool;

    /////// PARAMS //////////////
    int sampleRate = 44100;
    int frameSize = 2048;
    int hopSize = 1024;


Here we start by calling the ``essentia::init()`` function. If you forget to do that,
the algorithm factory will be empty and you won't be able to do much with Essentia!
We then create a Pool, and define some parameters we will use to configure our Algorithms.


Creating the required algorithms
--------------------------------

Here we create our Algorithms, configuring them on the fly. ::

  AlgorithmFactory& factory = standard::AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate);

  Algorithm* fc    = factory.create("FrameCreator",
                                    "frameSize", frameSize,
                                    "hopSize", hopSize);

  Algorithm* w     = factory.create("Windowing",
                                    "type", "blackmanharris62");

  Algorithm* spec  = factory.create("Spectrum");
  Algorithm* mfcc  = factory.create("MFCC");


It would have been equivalent to first create the Algorithms, and then configure them using a
``ParameterMap`` which would have been filled with these parameters.
However, it is much shorter (and cleaner, in the author's view) to write it like this.
When necessary, use `algorithm reference <algorithms_reference.html>`_ in order to get to know which parameters 
are required and what is their type and default values. 


Connecting the algorithms
-------------------------

Now we're mostly set to go, except that we're still missing something: the variables
in which the data will be stored. As you should know, inputs and outputs don't contain
the data they work on, but merely point to it. So we need to allocate this ourselves,
and tell the Algorithms' inputs and outputs to use these. ::


  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos ---------" << endl;

  // Audio -> FrameCutter
  std::vector<Real> audioBuffer;

  audio->output("audio").set(audioBuffer);
  fc->input("signal").set(audioBuffer);

  // FrameCutter -> Windowing -> Spectrum
  std::vector<Real> frame, windowedFrame;

  fc->output("frame").set(frame);
  w->input("signal").set(frame);

  w->output("windowedSignal").set(windowedFrame);
  spec->input("signal").set(windowedFrame);

  // Spectrum -> MFCC
  std::vector<Real> spectrum, mfccCoeffs, mfccBands;

  spec->output("spectrum").set(spectrum);
  mfcc->input("spectrum").set(spectrum);

  mfcc->output("bands").set(mfccBands);
  mfcc->output("mfcc").set(mfccCoeffs);



Processing the audio
--------------------

That's it, everything is in place, ready to be processed. We can now start calling
our algorithms' compute() functions. ::

  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  audio->compute();


This call will load all the audio data where the output of the ``audio`` algorithm
points to, that is, the ``audioBuffer`` variable. ::


  while (true) {

    // compute a frame
    fc->compute();

    // if it was the last one (ie: it was empty), then we're done.
    if (!frame.size()) {
      break;
    }

    // if the frame is silent, just drop it and go on processing
    if (isSilent(frame)) continue;

    w->compute();
    spec->compute();
    mfcc->compute();

    pool.add("lowlevel.mfcc", mfccCoeffs);

  }


Now, we loop over all the frames that the FrameCutter can get from the buffer that
has been set at its input (``audioBuffer`` again), and will write them at its output,
which points to the ``frame`` variable.

When the FrameCutter won't be able to output any more frame, it will output an empty one.
In that case, we should jump out of the loop, by the means of the ``break`` statement.

Next, we need to be careful in which order to call the functions. That is, at the moment
we only have a frame which is computed, so we first need to call the Windowing algorithm
so that it can window it. Calling the Spectrum first would only have computed the spectrum
from last frame again, as the data from the new frame hasn't arrived to its input yet.

To keep it simple, just make sure you call the ``compute()`` methods in the same order you
would write a block diagram explaining what you are doing.

Which gives us: ``Windowing::compute()``, then ``Spectrum::compute()``, then ``MFCC::compute()``.

At this point, we have the MFCCs computed for a frame and ready to be used. However, we
first want to compute them over all frames of the song, so we store them in the Pool, by
calling the ``Pool::add()`` method.


Aggregating the results and writing them to disk
------------------------------------------------

Now that we have computed the MFCCs for all the frames in our audio signal, we first want
to aggregate them::

  // aggregate the results
  Pool aggrPool; // the pool with the aggregated MFCC values
  const char* stats[] = { "mean", "var", "min", "max" };

  Algorithm* aggr = AlgorithmFactory::create("PoolAggregator",
                                             "defaultStats", arrayToVector<string>(stats));

  aggr->input("input").set(pool);
  aggr->output("output").set(aggrPool);
  aggr->compute();

This should be fairly straight-forward by now: instantiate and configure the algorithm,
set the inputs/outputs and call ``compute()``. Note here that algorithms can indeed take
any type of data as either input or output; in this case the input and output type of
data is a ``Pool``. ::


  // write results to file
  cout << "-------- writing results to file " << outputFilename << " ---------" << endl;

  Algorithm* output = AlgorithmFactory::create("YamlOutput",
                                               "filename", outputFilename);
  output->input("pool").set(pool);
  output->compute();


Writing the results is also done by the means of an Algorithm, although in this case
the algorithm doesn't have any output (writing to the file can be considered as a
side-effect, not the result of a pure function).

At this point, the only thing left to do is cleanup everything that we have used,
which is done in the following way::

  delete audio;
  delete fc;
  delete w;
  delete spec;
  delete mfcc;
  delete output;

  essentia::shutdown();

  return 0;


We delete all the algorithms that we created, and we also call ``essentia::shutdown()`` to
tell the library to free all the memory it might have allocated for itself. At this point,
it is safe to return 0 to the system, as should all well-behaved applications.

.. _imperative: http://en.wikipedia.org/wiki/Imperative_programming


Compiling extractors
--------------------

The simplest way to compile your own extractor is to place its code in ``src/examples`` folder and update 
the build script located in the same folder (``src/examples/wscript``). Add a new command similar to 
the ones already present in the script: ::

    build_example('standard', 'myextractorname')

Configure the build system to include compilation of examples if you have not done it before.::
    
    ./waf configure --mode=release --with-examples

Compile your examples by running::
    
    ./waf

See :doc:`installing` for compilation details.

