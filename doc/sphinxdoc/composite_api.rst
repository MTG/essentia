.. highlight:: cpp

Composite algorithm description
===============================

Composite algorithms are a special class of algorithms which allow you to create complex
algorithms using simpler ones. These can either be algorithms which are more complex
than others, and want to use more atomic operations (e.g.: MFCC would use the MelBands and
DCT algorithms), or can be entire extractors (e.g., the KeyExtractor would use the
following: FrameCutter → Windowing → Spectrum → SpectralPeaks → HPCP → Key).

The ``AlgorithmComposite`` class inherit the base ``streaming::Algorithm``, so you can use
them everywhere you would use a normal algorithm otherwise. There is however one big
advantage of composite algorithms over a normal encapsulation such as what you find in
object-oriented programming, which is that the scheduler knows about the internals of
composites, and as such you will keep all the parallelism and fine-grained execution you
would have by connecting all the constituent components of a composite, all of this while
giving you the nice API and ease-of-use of high-level signal processing blocks.

In order to achieve this, composites are not executed normally as the other algorithms,
for which the scheduler just call the ``process()`` method. Composite algorithms declare
the way they should be executed to the scheduler.

When defining an ``AlgorithmComposite`` you will have to implement the pure virtual
``declareProcessOrder()`` method.

In this method, you will specify a list of execution steps that need to be followed in
sequential order, and which can be either one of two possible steps:

* ``ChainFrom(algo)``: which will run the given algorithm and all algorithms connected
  to it (and contained in the composite) as if they were an entire network
* ``SingleShot(algo)``: which will run a given algorithm once

Each step will depend upon the complete execution of the previous one, so if you have a
small network that branches inside your composite, it will be better to run it using
ChainFrom(root) rather than SingleShot(algo) on all algorithms, as this will keep any
possible parallelization of the branches.

Example
-------

Here is the schema of a simplified KeyExtractor in the context of a larger network 
of algorithms, specified by a user.

.. image:: _static/essentia_tonal_extractor_modes_halfsize.png

You can see that although the user just connected the MonoLoader to the KeyExtractor,
and the KeyExtractor to the pool, the underlying network is a bit more complex.

Other things we can note:

* you can nest any number of times a composite into another one. This should be obvious
  due to the fact that a composite algorithm is itself an algorithm, so they can be used
  everywhere a "normal" algorithm can. For instance, here we have the Key algorithm
  (composite) used inside the KeyExtractor composite.

* composite algorithms need to have ``Sinks`` and ``Sources`` of their own, but in fact
  they just act as relay to/from the source/sink of one of their inner algorithms.

  This is done using the ``SinkProxy`` / ``SourceProxy`` class. They work in the same way
  as normal sources and sinks do, and you also have to declare them when building your
  algorithm. However, you will then need to attach them to the actual source or sink of
  the inner algorithm.

  Note that the proxies do not need to have the same name as the connector they are attached to.


Here follows what would be an implementation for the previous example. (simplified version,
irrelevant parts of code dropped) ::

  class MonoLoader : public AlgorithmComposite {
   protected:
    Algorithm* _audioLoader;
    Algorithm* _mixer;
    Algorithm* _resample;

    SourceProxy<AudioSample> _audio;

   public:
    MonoLoader() {
      // declare our output normally, except we're using a SourceProxy here
      declareOutput(_audio, "audio", "the audio signal");

      // create our inner network
      AlgorithmFactory& factory = AlgorithmFactory::instance();

      _audioLoader = factory.create("AudioLoader");
      _mixer       = factory.create("MonoMixer");
      _resample    = factory.create("Resample");

      _audioLoader->output("audio")           >>  _mixer->input("audio");
      _audioLoader->output("numberChannels")  >>  _mixer->input("numberChannels");
      _mixer->output("audio")                 >>  _resample->input("signal");

      // now attach the output of our last algorithm to the proxy of the composite
      // to allow data to be relayed outside of the composite
      _resample->output("signal")  >>  _audio;
    }

    void declareProcessOrder() {
      declareProcessStep(ChainFrom(_audioLoader));
    }
  };



::

  class KeyExtractor : public AlgorithmComposite {
   protected:
    Algorithm *_frameCutter, *_windowing, *_spectrum, *_spectralPeaks, *_hpcp, *_key;

    SinkProxy<Real> _audio;
    SourceProxy<std::string> _keyKey;
    SourceProxy<std::string> _keyScale;
    SourceProxy<Real> _keyStrength;

   public:
    KeyExtractor() {
      // declare inputs/outputs
      declareInput(_audio, "audio", "the audio signal");
      declareOutput(_keyKey, "key", "see Key algorithm documentation");
      declareOutput(_keyScale, "scale", "see Key algorithm documentation");
      declareOutput(_keyStrength, "strength", "see Key algorithm documentation");

      // instantiate all required algorithms
      _frameCutter   = factory.create("FrameCutter");
      _windowing     = factory.create("Windowing", "type", "blackmanharris62");
      _spectrum      = factory.create("Spectrum");
      _spectralPeaks = factory.create("SpectralPeaks",
                                      "orderBy", "magnitude", "magnitudeThreshold", 1e-05,
                                      "minFrequency", 40, "maxFrequency", 5000, "maxPeaks", 10000);
      _hpcpKey = factory.create("HPCP");
      _key     = factory.create("Key");

      // attach input proxy(ies)
      _audio  >> _frameCutter->input("signal");

      // connect inner algorithms
      _frameCutter->output("frame")          >>  _windowing->input("frame");
      _windowing->output("frame")            >>  _spectrum->input("frame");
      _spectrum->output("spectrum")          >>  _spectralPeaks->input("spectrum");
      _spectralPeaks->output("magnitudes")   >>  _hpcpKey->input("magnitudes");
      _spectralPeaks->output("frequencies")  >>  _hpcpKey->input("frequencies");
      _hpcpKey->output("hpcp")               >>  _key->input("pcp");

      // attach output proxy(ies)
      _key->output("key")       >>  _keyKey;
      _key->output("scale")     >>  _keyScale;
      _key->output("strength")  >>  _keyStrength;
    }

    void declareProcessOrder() {
      declareProcessStep(ChainFrom(_frameCutter));
    }
  };


And here is the code for the ``Key`` algorithm. As you will see, this one is a bit different
than the previous ones. Let's have a look at it first::


  class Key : public AlgorithmComposite {
   protected:
    SinkProxy<std::vector<Real> > _pcp;

    Source<std::string> _key;
    Source<std::string> _scale;
    Source<Real> _strength;

    Pool _pool;
    Algorithm* _poolStorage;
    standard::Algorithm* _keyAlgo;

   public:
    Key() {
      declareInput(_pcp, "pcp", "the input pitch class profile");
      declareOutput(_key, 0, "key", "the estimated key, from A to G");
      declareOutput(_scale, 0, "scale", "the scale of the key (major or minor)");
      declareOutput(_strength, 0, "strength", "the strength of the estimated key");

      _keyAlgo = standard::AlgorithmFactory::create("Key");
      _poolStorage = new PoolStorage<std::vector<Real> >(&_pool, "internal.hpcp");

      _pcp  >>  _poolStorage->input("data");
    }

    void declareProcessOrder() {
      declareProcessStep(SingleShot(_poolStorage));
      declareProcessStep(SingleShot(this));
    }

    bool process() {
      // we only want to output a Key estimate at the end of our stream
      if (endOfStream()) {
        // here, we want to call the std algo on the mean of the pcp frames
        vector<Real> hpcpMean = meanFrames(_pool.value<vector<vector<Real> > >("internal.hpcp"));
        string key, scale;
        Real strength;
        _keyAlgo->input("pcp").set(hpcpMean);
        _keyAlgo->output("key").set(key);
        _keyAlgo->output("scale").set(scale);
        _keyAlgo->output("strength").set(strength);
        _keyAlgo->compute();

        // now we have our values, push them out of the streaming algorithm
        _key.push(key);
        _scale.push(scale);
        _strength.push(strength);
      }
    }
  };


So, what can we see here:

 * the sources are actually ``Sources``, not ``SourceProxies``
 * the ``declareProcessOrder()`` method declares a process step on the algorithm itself
 * the ``process()`` method is actually defined here, alongside the ``declareProcessOrder()``

What happens here? Why are both the ``process()`` and the ``declareProcessOrder()``
methods defined?

What is actually happening is that the part calling the std version of the key algorithm
is quite small, and wouldn't warrant the creation of a new algorithm just for this purpose.
So we decided to keep this inside the Key algorithm, as if it were not a composite. This is
also the reason why the sources are actually ``Sources``, as we need them for
pushing the data through. If we had ``SourceProxies`` here, we wouldn't know where to attach them.

The following happens when the scheduler tries to run the Key algorithm:

* the scheduler wants to run the Key algorithm, it is a composite
* it looks at ``Key::declareProcessOrder()``; this contains 2 steps:

  1. ``SingleShot(_poolStorage)`` : fine, this is a normal call to _poolStorage.process(),
     which it executes
  2. ``SingleShot(this)`` : this is a recursive call of the Key algo, so the scheduler
     knows it shouldn't look at ``declareProcessOrder()`` now, but rather execute it
     normally, that is call the ``Key::process()`` method.

* the scheduler then goes on with the following algorithms

This way, there is no infinite recursion and everything is well-behaved. This might look a hack at first
sight, but actually it is a rather powerful mechanism that allows composite algorithms to be
more than just "chains" of other algorithms. Instead, they can be a mix of those chains and
specific code, without requiring this specific code to be artificially encapsulated in some
proxy algorithm.

Scheduling algorithm
--------------------

For more details on the inner workings of the scheduler, please have a look at the :doc:`execution_network_algorithm` page.
