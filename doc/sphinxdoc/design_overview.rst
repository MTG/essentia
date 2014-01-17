.. highlight:: c++

General architecture / Basic types and classes
==============================================

Essentia's main purpose is to serve as a library of signal-processing blocks.
As such, it is intended to provide as many pre-written algorithms as possible,
while trying to be as little intrusive as possible. Hence the general design is
very simple and easy to understand.

Each processing block is called an Algorithm, and has 3 differents types of attributes:

* Inputs
* Outputs
* Parameters

Every algorithm can have any number of each of these (zero included).
For instance, a "Centroid" algorithm will have 1 input (an array), 1 output (the value
of the centroid) and 1 parameter (the range of the centroid).

Basically, that is all you need to grasp what an Algorithm is in Essentia.

The general workflow is the following:

1. you instantiate (create) an algorithm
2. you configure it using the desired parameters
3. you feed it some input(s), and get back the output(s)
4. repeat 2. and/or 3. as much as desired

In the next sections, we will delve a bit more into the details.



Algorithms
----------

You already know that an Algorithm has inputs, outputs and parameters. To be more
precise, an ``essentia::Algorithm`` actually is a subclass of ``essentia::Configurable``,
which is the part that takes care of the parameters, and it adds the inputs and
outputs on top of it.

An algorithm is required to be able to perform a certain number of things (on top
of what being a Configurable requires, see :ref:`configurables`)

* you can get access to its inputs/outputs given their names
* you can ``compute()`` the result(s), that is, apply the specific algorithm to the
  inputs you have previously set, and get back the corresponding results
* if the algorithm maintains a state, you can ``reset()`` it.

Algorithms are stored in an ``essentia::AlgorithmFactory``, which contains information
about them and also knows how to instantiate them, using the
``essentia::AlgorithmFactory::create(const std::string&)`` method.

Source documentation: :essentia:`standard::Algorithm` and :essentia:`streaming::Algorithm`



Inputs / outputs
----------------

Inputs and outputs can be of any type in Essentia, and you can even create algorithms
in Essentia that use your own types. They are named, and in order to be recognized by
the algorithm they pertain to, they need to be declared explicitly using the
``essentia::Algorithm::declareInput()`` and ``essentia::Algorithm::declareOutput()`` methods.

It is highly recommended (although not mandatory) to declare them in the constructor of
your class (see :doc:`extending_essentia` for more details on how to write your own Algorithm).

Inputs and Outputs do not store data at all, they just point to it. So before calling
``compute()`` on an Algorithm, you need to make sure that its inputs/outputs point
to a correct place. To do this, you need to tell the input/output which variable
it should read/write in, by using the ``set()`` method.

For instance, to call a Centroid algorithm, you would do the following::


  std::vector<Real> myArray;  // the input array
  Real myCentroid;            // the variable containing the resulting centroid

  // point the input/output of the algorithm to their respective variable
  centroidAlgo->input("array").set(myArray);
  centroidAlgo->output("centroid").set(myCentroid);

  // only now can you call compute()
  centroidAlgo->compute();


.. _configurables:

Configurables
-------------

The ``Configurable`` class is the base class for the ``Algorithm``. A ``Configurable`` instance is
a named object that can maintain a fixed set of parameters, and which you can reconfigure any
number of times. To be able to instantiate a ``Configurable``, you need to implement the
``essentia::Configurable::declareParameters()`` method, which will declare all the
parameters that your Configurable object can take. If you later try to configure it
with a parameter that wasn't declared in the ``declareParameters()`` method, it will fail.

You can access the current value of a parameter by calling the
``essentia::Configurable::parameter(const std::string& name)`` method and passing it the
name of the parameter.

To (re)configure a Configurable, you need to call the ``configure(const ParameterMap& pmap)`` method.
This will check whether the parameters are acceptable, set them, and call the ``configure()`` method,
which you should have redefined if you want your object to do some specific action when being configured.

Source documentation: :essentia:`Configurable`


Parameters
----------

A Parameter is a variant type, meaning that it can basically represent any type of data.
For instance, at the moment of this writing, Parameters can represent strings, integers,
floating point numbers, booleans, vectors of strings or reals. More type conversions can
be added if necessary.

This is especially useful in C++ as it is a statically-typed language, but we want to allow
different types of data for configuring an algorithm. In Python, the point of having variant
types is moot, thanks to the dynamic typing.

Here is a small example of creating / retrieving the values of some parameters::

  Parameter param1(23);
  int param1_int = param1.toInt();

  std::vector<Real> v; // v is empty
  v.push_back(1.2);    // v = [ 1.2 ]
  v.push_back(2.3);    // v = [ 1.2, 2.3 ]
  Parameter param2(v);
  std::vector<Real> param2_vector = param2.toVectorReal();

  // conversions between types are allowed as long as they make sense
  Parameter param3(117);     // constructed from an integer
  Real p3 = param3.toReal(); // works because an integer is also a float


Another closely related class to ``Parameter`` is the ``ParameterMap``, which is just
a map from ``std::string`` (the name of the parameter) to ``Parameter`` (its value).
It represents a set of Parameters, and is mostly used in the call to the
``Configurable::configure(const ParameterMap& pmap)`` method.

Source documentation: :essentia:`Parameter` and :essentia:`ParameterMap`



Pool
----

A ``Pool`` is a thread-safe structure that is used to store values. It could be thought of as a cache.
Basically, during processing you generate lots of values which you want to post process
afterwards, and in that case, a ``Pool`` is the perfect candidate for a storage mechanism.

The pool stores these values using a ``std::string`` as identifier, which can be
dot ('.') separated to indicate namespaces. For instance, the following are all valid
names: ``filename``, ``lowlevel.centroid``, ``highlevel.genre.value``,
``highlevel.genre.rock.probability``, ...

There are 2 ways to store values in a pool: you can either ``add()``, or ``set()`` them.
When you add a value, it gets appended to the list of values with the same name, when you
set it, you replace the value which was previously stored with this name (or create it).
To retrieve those values, you need to call the ``value()`` function, which is templated by
the type of the value.

For instance, you might want to store all the values of the per-frame energy, and compute
the mean at the end to have an idea of the average energy for a track.

You could do it this way::

  Pool pool;
  while (moreFrames) {
    // compute energy here
    pool.add("lowlevel.energy", energyValue);
  }

  const vector<Real>& allEnergyValues = pool.value<vector<Real> >("lowlevel.energy");
  Real averageEnergy = mean(allEnergyValues);
  pool.set("highlevel.average_energy", averageEnergy);

  cout << "The average energy is: " << pool.value<Real>("highlevel.average_energy");


Note that although you feed the pool with a ``Real`` value for the energy, the call to
``Pool::value()`` will return a ``std::vector<Real>``, because it will return **all**
the values that you gave it. Even if you only added one value into the pool, a call
to ``value()`` will return a vector, of size 1 in that case.
On the other hand, if you used ``set()``, the value returned is of the same type.

Source documentation: :essentia:`Pool`


Logging framework
=================

Logging in C++
--------------

Essentia provides you with a logging framework that is meant to be both efficient
and easy to use. It tries to learn from other logging frameworks and has 4 logging
levels that can be activated/deactivated independently at runtime:

* Error
* Warning
* Info
* Debug

Furthermore, the debug level is itself subdivided into different debugging modules,
defined in the :essentia:`DebuggingModule` enumeration which you can find in the
:essentia:`debugging.h` file.

The way to debug is to use the following macros::

    E_ERROR("This is an error message!");
    E_WARNING("This is a warning message...");
    E_INFO("And this is an info message.");
    E_DEBUG(EMemory, "This is a debug message relating to the memory usage");
    E_DEBUG(EAlgorithm, "And this one is a debug message relating to an algorithm");

You can also log more than just a string using the stream operator, as you would do
in a C++ ``std::ostream``::

    E_INFO("You can log ints, such as " << 42 << " floats, as in " << 3.14 << " and " <<
           "pretty much anything that you can send in a std::ostream");

To activate/deactivate the debugging modules at runtime, use the functions::

    setDebugLevel(EAll);                     // EAll is a special value that contains all modules
    unsetDebugLevel(EMemory | EConnectors);  // modules are bitmasks

Note that when a logging module is deactivated, the cost on runtime is minimal
(i.e., you only really pay for logging when you use it). If you wish to completely
turn off logging, this can be done at compile time by setting the ``DEBUGGING_ENABLED``
variable in the :essentia:`config.h` file to ``0``. Note that in this case, it will not be
possible to activate *any* logging at runtime at all, the advantage being that you
pay absolutely nothing for logging, so don't hesitate to (ab)use logging in your
algorithms because of fear of losing efficiency.

Advanced logging
^^^^^^^^^^^^^^^^

Sometimes, you wish to only activate logging for a certain period of time, to avoid
being overwhelmed by too many messages. For example, if your algorithm fails at some point
in an audio file, you might want to only log some info for the 5 previous calls to the
generator (frames of the audioloader, usually).

This is done by specifying a list of time ranges (in number of calls to the generator) and
modules that should be activated during this range. For example::

    DebuggingSchedule s = { {0,   INT_MAX, ENetwork},     // always active
                            {500, INT_MAX, EAlgorithm },  // from time index 500 until the end
                            {782, 782,     EScheduler} }; // only for time index 782
    scheduleDebug(s, ARRAY_SIZE(s));

This will run a network and activate the ENetwork level always, the EAlgorithm
from the 500th call to the generator until the end, and the EScheduler
level only on the 782th call to the generator.


Logging in python
-----------------

In python, you would usually log things using the standard ``logging`` module, but
Essentia also gives you access to its C++ logger in order to ensure that
the output is not mangled between the 2 logging frameworks:

.. code-block:: python

    essentia.log.info("This is how you use Essentia's logger in python")
    essentia.log.debug(essentia.EAlgorithm, 'Debugging modules are also available')

    essentia.log.infoActive = True                   # activate the info level
    essentia.log.debugLevels += essentia.EAll        # activate all debug modules
    essentia.log.debugLevels -= essentia.EExecution  # deactivate the ``Execution`` one
