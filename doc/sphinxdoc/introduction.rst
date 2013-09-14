Introduction
============

Highlevel overview
------------------

Essentia is a library providing tools for performing analysis of audio data.

The main design factors are (in this order):

 - correctness of the implemented algorithms
 - ease of use
 - tidiness and maintainability of the code
 - performance
 - extensive collection of algorithms

Essentia has been developed in the context of Music Information Retrieval at
the `Music Technology Group <http://mtg.upf.edu>`_, and
tries to cater to several needs that a research centre faces to perform its
activities:

 - provide an interactive environment for researchers that facilitates
   experimentation and rapid application development
 - compile optimized extractors to be able to run them efficiently on computing
   clusters in order to analyze big databases of audio tracks

The performance requirement dictates that Essentia be written in C or C++, and
C++ has been chosen as it also allows to have a nicer object-oriented API, which
fits nicely in the context of audio processing.

Python bindings are also provided in order to be able to use Essentia in an
interactive development environment, and they fit naturally
with the IPython/NumPy/Matplotlib ecosystem (similar to Matlab).

In order to further facilitate writing feature extractors, Essentia also sports
a "streaming" mode, which allows to more easily connect algorithms together
in an intuitive way and with little boilerplate code required.

The following table shows the combination of C++/Python and the standard/streaming
modes and their recommended use cases.


+---------------------+---------------------------------+---------------------------------+
| mode \\ language    |  **C++**                        | **Python**                      |
+---------------------+---------------------------------+---------------------------------+
| **standard**        | - maximum control               | - interactive environment,      |
|                     | - requires more work to write   |   useful for research           |
|                     |   extractors                    |                                 |
+---------------------+---------------------------------+---------------------------------+
| **streaming**       | - easily write extractors       | - easily write extractors       |
|                     |                                 | - porting to C++/streaming is   |
|                     |                                 |   straightforward               |
+---------------------+---------------------------------+---------------------------------+

As a next step, it is recommended to read the `Algorithms overview`_ page
to get a feeling of the different types of algorithms available. You can then
continue to the `Python tutorial`_ page for a more hands-on introduction to using
Essentia in practice, with the python bindings for the essentia standard mode.

C++ developers can look at the `"Standard" mode how-to`_
and `"Streaming" mode how-to`_ pages to get
more familiar with Essentia.



Essentia history / Changelog
----------------------------

This section is a narrative that describes the evolution of Essentia since its
inception. It shows the needs for carrying audio analysis tasks and how they
defined Essentia's architecture.

The Essentia project was started in 2005 as a library in order to replace what
was at the time an organic collection of small ad-hoc c++ programs used for diverse
audio analysis tasks.

The 0.x releases continously refined the main concepts of Essentia bit by bit and
added algorithms in order to provide an extensive toolbox, all of which culminated
with the release of Essentia 1.0 in 2008


**Essentia 1.0**
^^^^^^^^^^^^^^^^

*(released April 2008)*

Essentia 1.0 was the first real "stable" release,
that defined the basic concepts of algorithms, sources/sinks, and
audio/numeric/string data types.

There are 2 use cases that Essentia claims to cover:

 - the research task: which is usually carried by a researcher in MIR for instance,
   in an interactive environment (think Matlab), where the main needs are
   ease-of-use and interactivity
 - the implementation task: that assembles those "research scripts" into programs
   that lose the interactivity but in return gain a potentially much faster run
   time, more suited to heavy analysis on big databases/computing clusters

The implementation task required that algorithms always be implemented in the
most efficient manner, in C++ in this case. In standard mode, the algorithms
never allocate data themselves and work in-place on the data provided by the
user. This ensures that the minimum amount of data can be copied at each time,
and most of the time no data copy at all is required in order to move data from
one algorithm to another.

We also introduced at this point the concept of standard/streaming mode, which
allow to either use the algorithms in an imperative way (which used only the
minimum resources possible, as allocated by the caller, but which requires the
complete enumeration in source code of all the actions to be taken), or to use
them in a data-flow oriented way which would take care automatically of all the
data transfer between algorithms, and ensure that buffers would be dealt with
appropriately (no buffer underrun/overrun)

This was done as a goal of making it as easy as possible to write feature
extractors. A tiny loss of performance as well as a little less control over
when the algorithms are processed were easily offset by the huge amount of time
that was saved when writing extractors. The fact that it does require less
boilerplate code also made it much less error-prone to write extractors
(synchronizing the computation of a dozen or more algorithms and moving data
between them is not that easy to do properly).

One can see what the streaming mode looks like in the `"Streaming" mode how-to`_,
or look at the `Streaming Architecture`_ to delve more into the technical details
of how that works.


To ensure that there is as little duplication of code as possible, we tried to
always have either the streaming mode implementation be a wrapper of the
standard mode one, or the other way around. This could be done for nearly all
the algorithms, except a few ones that were very specific to either mode of
operation (the infamous FrameCutter comes to mind...). This ensures that
algorithms always give the same result in standard and streaming mode, and
no maintenance burden is added over what would be there for only one mode of
operation.

To further ensure that we reuse code as much as possible, we introduced the
concept of AlgorithmComposite, to enable to easily define algorithms in the
streaming mode by "assembling" smaller ones together.
The `Composite API <composite_api.html>`_ page looks at this more in details.

Finally, as C++ was clearly not adapted to the interactivity needs from researchers,
Python bindings have also been provided for the standard mode of Essentia.
Python bindings for the streaming mode being more complex to implement, they were
dropped in favor of a specially designed `Domain-specific language`_ for defining
networks of algorithms, dubbed ESX (ESsentia eXtractor).

.. _Domain-specific language: http://en.wikipedia.org/wiki/Domain-specific_language


Essentia 1.0.x
^^^^^^^^^^^^^^

During the 1.0.x cycle, we performed a complete white box review of all the
algorithms available at that time, to ensure that APIs and naming were consistent,
that all algorithms had proper documentation (including scientific references),
and that they were also performing as intended, which lead to the writing
of ~1000 unit tests.

This huge review cycle ended with the 1.0.6 version, released in March 2009.


Essentia 1.1
^^^^^^^^^^^^

*(released August 2009)*

ESX being too much of a maintenance burden in the long-run, and too limiting in the
types of possibilies it offered (it being a DSL there was no other issue possible,
hindsight is 20/20), it was decided to invest the time to implement the python
bindings for the streaming mode.

This is also the release that introduced the new ffmpeg audio I/O, which allowed
us to get audio from pretty much any source and in any format (even works
directly with youtube videos!)

More mid-level and high-level descriptors have also been added to complete the
list of available algorithms.


Essentia 1.2
^^^^^^^^^^^^

*(released April 2010)*

An algorithm that allows to apply Gaia transforms has been added, which allows
to train classification models (SVM, nearest-neighbor, ...) and run them as
an Essentia feature extraction algorithm.

A more diverse set of prebuilt extractors has been written, in order to provide
useful out-of-the-box extractor, for people more interested in machine learning
instead of feature extraction. (lowlevel features, rhythm, tonal/key, etc.)

A `Vamp plugin`_ has been written for some of the algorithms (mostly the low-level
ones).

.. _Vamp plugin: http://www.vamp-plugins.org/


Essentia 1.3
^^^^^^^^^^^^

*(released December 2011)*

This release introduced a new rhythm algorithm (BPM, beat detection) with
improved performance. Apart from that, it was mostly a maintenance release
with a lot of fixes.


Essentia 2.0
^^^^^^^^^^^^

*(released ??? 2013)*

This major release is the first release to be publicly available as free
software. It also features a refactoring of the core API, a little bit for
the standard mode to fix small API annoyances, but mostly for the streaming
mode which is now much better defined, using sound computer science
techniques.

In particular, the scheduler that runs the algorithms in the streaming mode
is now a lot more correct, which permitted to clean all the small hacks that
had accumulated in the algorithms themselves during the 1.x releases to
compensate for the deficiencies of the initial scheduler.



.. _Algorithms overview: algorithms_overview.html
.. _Streaming architecture: streaming_architecture.html
.. _Python tutorial: python_tutorial.html
.. _"Standard" mode how-to: howto_standard_extractor.html
.. _"Streaming" mode how-to: howto_streaming_extractor.html
