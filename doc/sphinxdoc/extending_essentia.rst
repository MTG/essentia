.. highlight:: cpp

Extending Essentia / Writing a new Algorithm
============================================

Although Essentia comes pre-loaded with a considerable list of algorithms, we imagine 
that you will want to write your own algorithms at some point. By following the next steps,
you should be able to roll up your own algorithm in very little time.

As an example, you should have a look at the implementation of the Centroid algorithm,
which you can find in the
:download:`src/algorithms/stats/centroid.h <../../src/algorithms/stats/centroid.h>`
and the
:download:`src/algorithms/stats/centroid.cpp <../../src/algorithms/stats/centroid.cpp>`
files.


Detailed steps for the creation of a new Algorithm
--------------------------------------------------

Declaring basic information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To write a new Algorithm, you will have to inherit from the ``essentia::Algorithm`` class.
The first thing that you will want to do is to give your algorithm a name and a description.
There are two fields specifically dedicated to this, which you *need* to fill in, otherwise
it will be impossible for you to register your algorithm in the factory
(it will give you a compilation error if you try)::

  // in your header
  class MyAlgo : public Algorithm {
    static const char* name;
    static const char* description;
  };

  // in your source file
  const char* MyAlgo::name = "MyAlgo";
  const char* MyAlgo::description = DOC("This is my new algorithm that does lots of stuff");


Note that the description should be surrounded by the ``DOC()`` directive.

Declaring/registering your inputs and outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To be able to pass data to/from your algorithm, you will then need to declare what goes
in and what goes out. These will take the form of any number of Inputs and Outputs, which
can be of any type you want. You will also have to assign names to each one of your input
and output. To do this, you will have to declare in your derived class member variables
that are wrapped by the ``Input<>`` and ``Output<>`` wrappers, and register them in the
constructor.

For instance, if you have an input that is an audio frame (represented using a
``std::vector<Real>``) and your output is the high-frequency coefficient (HFC)
(represented using a ``Real``), you would:

1. declare the following as member variables of your class::

      Input<std::vector<Real> > _audioFrame;
      Output<Real> _hfc;


2. register them in the constructor (otherwise they won't be visible), with their
   (compulsory) description::

      MyAlgorithm::MyAlgorithm() {
        declareInput(_audioFrame, "audio", "the input audio frame");
        declareOutput(_hfc, "hfc", "the resulting high-frequency coefficient");
      }

   The names that you use when registering the inputs/outputs are the names that will later be used
   to access them.


Declaring needed parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a function that you will always have to fill-in, which is the
``void declareParameters()`` method.
This is used to tell what are the parameters that are used by this algorithm, so that it
is able to make sure you will get all the parameters you need. For this, you will need to
declare the names and description of the parameters you are expecting, as well as an
optional default value. There are 2 methods you can use in order to do this::

  void declareParameter(const std::string& name, const std::string& description);
  void declareParameter(const std::string& name, const std::string& description, const Parameter& default_value);

The first version is to declare a single Parameter, the second one does the same but also
specifies a default value. An example of the declareParameters() method follows::


  void declareParameters() const {
    declareParameter("sampleRate", "the sampling rate of the analyzed track");
    declareParameter("nbCoeffs", "the number of coefficients to be output", 12);
    declareParameter("floatParam", "a random floating point parameter", 23.8);
    declareParameter("windowType", "the type of window used before doing the FFT", "Hann");
  }


Note: the ``Parameter`` type is sort-of dynamic, so when specifying a default value, you
can do it using its native type (i.e., int, float, string, ...) as the conversion to the
``Parameter`` type is done automatically.


The configure() method
^^^^^^^^^^^^^^^^^^^^^^

This method (with no parameters) will be called each time the object is configured.
This is intended if you have some setup actions to do before starting to process that you
only want to be done once (e.g., setting planes in an FFT, preparing cos&sin tables for MFCC
computation, etc...). As a rule of thumb, you can (and should) initialize everything you
can in the constructor (i.e., when not knowing any parameters) and initializes the rest of
it (that is dependent on parameters) in the configure method.

You will be given as input a ``ParameterMap`` containing all of the parameters that you 
declared using the ``declareParameters`` statements.


Checking if the object is configured
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For user convenience and more consistency, the ``configure()`` method will be
called with default values upon creation, so that the object is always configured. If you declared
some parameters in your ``declareParameters()`` method which do not have default values, when
creating the object there will be some parameters missing upon entering the ``configure()`` method.
You should make sure in that case that ``configure()`` still returns correctly and that the object
is not in an invalid state.

The reset method
^^^^^^^^^^^^^^^^

When doing batch computation (i.e., multiples files/sounds in a row), it might be usefull (or necessary)
to reinitialize your algorithm between different files if it keeps a state of itself. This is the purpose of the
``reset()`` method.

Note: most of the people won't need to use this, as the descriptor calculation won't have any state.


The compute method
^^^^^^^^^^^^^^^^^^

This is the main entry point for your ``Algorithm``. It is the generic function that is used to
tell an ``Algorithm`` to compute the things it is supposed to.
This method will be called once the inputs and outputs are set.
Basically the first thing you will want to do is get the inputs and outputs into local variables
and then do your processing. This is done through the ``get()`` method that is defined for both
the ``Input<>`` and ``Output<>`` classes, and it returns a reference to the type they are
wrapping. Inputs are const references, Outputs are non-const references, so you can write to them.

Example::

  Input<vector<Real> > _audio;
  const vector<Real>& audioVector = _audio.get();

  Output<string> _label;
  string& genreLabel = _label.get();


Notice that genreLabel is not const, so that you can write to it, ie::

  genreLabel = "Electro";


Another way to write your ``compute()`` method (and if the parameterless way of calling it
disturbs you), is to write your function in the 'classic' way, passing the inputs as arguments
to the function call, and then wrapping this call with the parameterless ``compute()`` method.

Example::

  void compute() {
    // inputs and parameters
    const vector<Real>& array = _array.get()
    Real frequencyRange = parameter("frequencyRange").asReal();

    // output
    Real& centroid = _centroid.get();

    // do the actual work
    centroid = centroid_function(array, frequencyRange);
  }

  Real centroid_function(const vector<Real>& array, Real frequencyRange) {
    // your implementation here
  }


**Note:** make sure that when using get, you **always** use references (&), and not a copy, otherwise

1. your outputs won't be stored
2. you'll be making unnecessary copies of your inputs, which can considerably slow down the execution time.


Here are some examples that you can have a look at to get you started:

- RMS *(easy)*
  (:download:`rms.h <../../src/algorithms/stats/rms.h>` and
  :download:`rms.cpp <../../src/algorithms/stats/rms.cpp>`)
- Resample *(medium)*
  (:download:`resample.h <../../src/algorithms/standard/resample.h>` and
  :download:`resample.cpp <../../src/algorithms/standard/resample.cpp>`)
- Trimmer *(medium)*
  (:download:`trimmer.h <../../src/algorithms/standard/trimmer.h>` and
  :download:`trimmer.cpp <../../src/algorithms/standard/trimmer.cpp>`)
