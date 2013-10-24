.. highlight:: c++

Essentia streaming mode architecture
====================================

In Essentia, the standard mode of operation is imperative: once you have created
your algorithm, you have to set its inputs and outputs, then call the ``compute()``
method yourself, as many times as needed. This also means that you have to take care 
about scheduling yourself, ie: if you want to compute the MFCCs of a file, you have 
to read the file, *then* slice it into frames, *then* loop over each frame and call 
*successively* the windowing of the frame, the FFT, and *finally* the MFCC computation.

In the streaming mode, you do not call the ``compute()`` method yourself anymore.
Instead, you define a network of connected algorithms, and an internal scheduler
takes care of calling the appropriate ``compute()`` methods for you, whenever it's
needed. Of course, this needs a bit more work on the developer side, but the user
side gets simplified a lot: the user doesn't have to care anymore about
*how* and *when* to call the ``process()`` method, but can focus on *what* he wants to
compute and leave the rest to the computer. You can think of it as the data-flow mode, i.e., 
the data flows between algorithms automatically in accordance with the defined network. 


Anatomy of a StreamingAlgorithm
--------------------------------

As in the standard mode, a ``streaming::Algorithm`` only consists of a few specific
concepts that you need to understand. These are:

* it inherits from ``Configurable``
* it uses the ``Sink`` and the ``Source`` classes, instead of ``Input`` and ``Output``
* the ``process()`` method replaces the ``compute()`` method

Apart from that, you might need to delve into advanced concepts, such as how are
algorithms scheduled to run, or how does the consumption model work. 
It is not always necessary, however, and in most of the cases you can simply use 
wrappers and predefined functions.


Connecting it all together, creating a network
----------------------------------------------

To compute descriptors in the streaming mode, you will need to choose the algorithms
you want to be called, and connect them together using their sinks/sources.
This will create a network.

Sources and sinks are typed connectors that allow you to connect algorithms together.
You can only connect a source to a sink, and you can only connect these if they are of the same type.

For example, you cannot connect a ``Source<Real>`` to a ``Sink<vector<Real> >``, because
they don't have the same type. If you want to connect an audio source (``Source<Real>``)
to an algorithm that computes the FFT (``Sink<vector<Real> >``), you will first need to
put a ``FrameCutter`` in between, that will transform the flow of ``Real`` (audio samples)
into a flow of ``vector<Real>`` (frames).

Connecting a Source to a Sink means that all the data that comes out of the Source
will be forwarded to the Sink. 

To connect a source to a sink, you have to use ``connect(Source& source, Sink& sink)`` function
or, completely equivalent, the ``>>`` operator::

  connect(algo1->output("abc"), algo2->input("def"));

  // or, exactly the same, but looks nicer
  algo1->output("abc")  >>  algo2->input("def");


You can connect a single Source to multiple Sinks (i.e., you can have more than one algorithm
reading the output of your source), but you cannot connect more than one Source to a single
Sink (i.e., a Sink can only get its data from a single Source, there is no automatic multiplexing
or anything like that (except if you use a specific algorithm)).

There is also one more restriction at the moment: the network you build must be a connected
acyclic graph (i.e., a tree). **Warning:** if you have different branches that later merge again (e.g., if
you have a diamond-shape in your graph), it might still work but you have absolutely no
guarantee of correctness, nor that it will not crash or anything bad like that.

The fact that your network is a tree means that there is a root node, which will be the
algorithm feeding data to other algorithms. This node is called a generator, and most
of the time will be an audio loader.


Special connectors
------------------

When connecting your algorithms together, there is one rule which you should never break:
**All sources and sinks must be connected.**

However, in certain cases the connection you need to make might not be obvious and you should use
specific connectors for these cases. For instance, a *NOWHERE* connector should be used to ignore 
an output of an algorithm you do not want to use: when connected to it, all the data coming out of 
the source will be discarded.


These are the specific connectors you can use in Essentia:

* the *NOWHERE* connector (also called *DEVNULL*), which discards the data coming from the
  source it is connected to ::

    connect(audioLoader->output("audio"), NOWHERE);


* the Pool connector, which stores the values coming out of the stream inside the Pool with
  the specified namespace and name ::

    connect(centroidAlgo->output("centroid"), PoolConnector(pool, "lowlevel.centroid"));

    // or, more succinctly, using the "PC" typedef
    centroidAlgo->output("centroid")  >>  PC(pool, "lowlevel.centroid");

* the VectorInput / VectorOutput auto-connectors, which allow you to connect a ``std::vector``
  directly to a ``Sink`` or a ``Source`` directly to ``std::vector`` ::

    std::vector<Real> inputVector, outputVector;

    // note that for this to work, input has to be a Sink<Real> and output a Source<Real>
    connect(inputVector, algo->input("data"));
    connect(algo->output("result"), outputVector);

* the FileOutput connector, which is an Algorithm to which you connect directly instead
  of connecting to its input sink, and which is "dynamically-typed" (sort of), meaning you
  can connect any source type to it ::

    Algorithm* output1 = factory.create("FileOutput",
                                        "filename", "out1.txt");
    Algorithm* output2 = factory.create("FileOutput",
                                        "filename", "out1.txt");

    // algo1->output("x") is a Source<Real>
    connect(algo1->output("x"), *output1);

    // algo2->output("y") is a Source<std::vector<std::string> >
    connect(algo2->output("y"), *output2);



Scheduling
----------

Creating a network
^^^^^^^^^^^^^^^^^^

Once you have connected all your algorithms together, you need to declare that they form a network 
by instantiating an ``essentia::scheduler::Network``. A Network is created from
the generator node (an audio loader most of the time)::

  Algorithm* audioLoader = factory.create("MonoLoader",
                                          "filename", "test.mp3");
  Algorithm* extractor   = factory.create("Extractor");

  audioLoader->output("signal")  >>  extractor->input("data");

  // here we create our network
  scheduler::Network network(audioLoader);


The Network takes ownership of the algorithms and knows how to dispatch commands to
them, for instance, if you want to reset all the algorithms contained in the network
you would call ``Network::reset()``, and when the network goes out of scope it will also
take care of deleting all the algorithms it contains.



Starting a network
^^^^^^^^^^^^^^^^^^

The expected behavior of a ``streaming::Algorithm`` is that as soon as there is enough
data to be processed, it will consume the data at its input(s), process it, and produce
the result at its output(s).
The only exception to this rule are the generators, which continuously produce data
on their output(s).

Thus, when you have a network ready, you simply have to tell it to start processing 
data. The internal scheduler will take care of delivering the data wherever it needs to,
taking also care that there are no congestion and that all algorithms get executed when they should be.

This is done very easily by calling the ``Network::run()`` method::

    scheduler::Network network(audioLoader);

    // run it!
    network.run()

This function will only return once all the data that the generator could produce
has successfully flown through all the connected algorithms.


Implementation details
----------------------

In this part, we will examine a bit more in details how the data flows between sources and
sinks. It is highly recommended to read this paper, as some concepts and vocabulary are heavily borrowed from it:

http://hillside.net/plop/2006/Papers/Library/audioPatterns_20060809.pdf

If you don't want to read the whole paper, you should make sure that you know and
understand at least the following concepts: token, typed connections, multiple window circular buffer.

There is one notable difference with the aforementioned article, which is that in-ports
(resp. out-ports) are called sinks (resp. sources) in Essentia.

In Essentia, Sources and Sinks are implemented as templates, so a sink of real values
would be declared like this::

  Sink<Real> mySink;


``StreamingAlgorithm``\ s are conceptually the same as standard ``Algorithm``\ s, except
that you need to replace the ``Input``\ s and ``Output``\ s by ``Sink``\ s and ``Source``\ s.
They are still configurable in the same way as ``Algorithm``\ s were.

Consumption model
^^^^^^^^^^^^^^^^^

For performance reasons, Essentia uses ``PhantomBuffers``, but for the sake of clarity,
let's assume that they are simple circular buffers, with one writing window (e.g., one producer,
the source) and many reading windows (e.g., multiple consumers, the sinks connected to this source).
Please see the aforementioned paper for more details on the multiple window circular buffer pattern.

As consuming/producing data is independent of whether we're talking about a ``Sink`` or a
``Source``, this functionality is factored into the ``essentia::StreamConnector`` class,
which both ``Source`` and ``Sink`` inherit.

Consumption/production is done in the following way:

* you first need to acquire some tokens (which will expand your window) by calling
  ``StreamConnector::acquire(n_tokens)``. This method will succeed if and only if there were
  enough available tokens in the buffer.
* if that call succeeded, you will then do the processing you intended to do with these.
* you can then call ``StreamConnector::release(n_tokens)``, which will then yield these tokens back.

This is very general and works the same for both sinks and sources. Here is a somewhat more detailed 
explanation for both cases:

for a ``Source``:

* ``Source::acquire()`` makes sure there are enough tokens to write in the buffer and reserves them.
* ``Source::release()`` actually produces them, and they become available to the readers.
  Remember that before this call, due to the fact that no reader windows can overlap over the
  writing window, these tokens were not yet accessible.

for a ``Sink``:

* ``Sink::acquire()`` takes n tokens and makes sure they stay available during processing.
* ``Sink::release()`` yields back reservation of these tokens so they can be written again when
  the writing window does a full cycle.


Please note that you can release a different number of tokens than what you acquired without
any problem. An algorithm that implements a moving average of the last ``n`` samples would
acquire ``n`` samples each time (the size of the window), but only release 1 sample each time
(advance the window by 1 sample).

A signal processing analogy that might help you better visualize this is:
``acquire(int n_tokens)`` takes the **frame size** of the data you want to analyze as argument,
while ``release(int n_tokens)`` takes the **hop size** as argument.
