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

Essentia has been developed in the context of research activities in Music
Information Retrieval that were held at the `Music Technology Group <http://mtg.upf.edu>`_.
It caters for the needs of both rapid prototyping and large-scale analysis, in particular:

 - to provide an interactive environment for researchers that facilitates
   experimentation and rapid application development
 - to compile optimized extractors to be able to run them efficiently on computing
   clusters in order to analyze large databases of audio tracks


The library is developed in C++ in order to ensure high performance and provide an 
object-oriented API that suits audio processing. Python bindings are also provided in 
order to be able to use Essentia in an interactive development environment, and they fit naturally
with the IPython/NumPy/Matplotlib environment (similar to Matlab).

In order to further facilitate writing feature extractors, Essentia also suppports
a "streaming" mode, which allows to connect algorithms together more easily
in an intuitive way and with little boilerplate code required.

The following table shows the recommended use-cases for the usage of C++/Python in
combination with the standard/streaming mode.


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
to familarize with the different types of available algorithms. You can then
continue to the `Python tutorial`_ page for a more hands-on introduction to using
Essentia in practice, with the python bindings for the Essentia's standard mode.

C++ developers can look at the `"Standard" mode how-to`_
and `"Streaming" mode how-to`_ pages to get
more familiar with Essentia.



Essentia history / Changelog
----------------------------

This section is a narrative that describes the evolution of Essentia since its
inception. It shows how the  needs for carrying audio analysis tasks defined Essentia's 
architecture.

Essentia library was started in 2006  in order to replace a collection of small 
ad-hoc C++ programs developed at Music Technology Group and used for diverse audio analysis 
tasks. The 0.x releases continuously refined the main concepts of Essentia bit by bit and
added algorithms in order to provide an extensive toolbox, all of which culminated with 
the release of Essentia 1.0 in 2008.


**Essentia 1.0**
^^^^^^^^^^^^^^^^

*(released April 2008)*

Essentia 1.0 was the first real "stable" release, that defined the basic concepts 
of algorithms, sources/sinks, and audio/numeric/string data types.

The design of Essentia was defined by the two use-cases:

 - the research task: which is usually carried by a researcher in MIR for instance,
   in an interactive environment (think Matlab), where the main needs are
   ease-of-use and interactivity
 - the implementation task: that assembles those "research scripts" into programs
   that lose the interactivity in return for a potentially much faster run
   time, more suited for heavy analysis on large databases/computing clusters

The implementation task required algorithms to be implemented in C++ in the most efficient 
manner. A concept of standard and streaming modes was introduced in order to allow 
the use of algorithms either in an imperative way or a data-flow oriented way. 

In standard mode, the algorithms never allocate data themselves and work 
in-place on the data allocated by the caller. This ensures that the minimum amount of 
data is copied at each time, and most of the time no data copy at all is required 
in order to move data from one algorithm to another. In order to implement a chain of 
algorithms, standard mode requires a user to state in the code all the required actions to be taken. 
In contrast, streaming mode allows for automatic data transfer between algorithms 
and ensures correct treatment of the buffers (no buffer underrun/overrun). 

Streaming mode was introduced with the goal to make implementation of feature extractors 
(typically containing large processing chains of Essentia algorithms) 
as easy as possible. A tiny loss of performance as well as a little less control over
when the algorithms are processed were easily offset by the huge amount of time
that was saved when writing extractors. In addition, the fact that it does require less
boilerplate code made the process of writing extractors much less error-prone
(synchronizing the computation of a dozen or more algorithms and moving data
between them is not that easy to do properly).

One can familiarize with the streaming mode in the `"Streaming" mode how-to`_,
or look at the `Streaming Architecture`_ to delve more into the technical details
of how it works.


To ensure that there is as little duplication of code as possible, we tried to
always have either the streaming mode implementation be a wrapper of the
standard mode one, or the other way around. This could be done for nearly all
the algorithms, except a few ones that were very specific to either mode of
operation (e.g., FrameCutter). This ensured that algorithms always give the 
same result in standard and streaming mode, and no maintenance burden is added 
over what would be there for only one mode of operation.

The concept of AlgorithmComposite was introduced to further ensure that the code 
is reused as much as possible. It enabled to easily define algorithms in the
streaming mode by "assembling" smaller ones together. The `Composite API <composite_api.html>`_ page 
provides details on using AlgorithmComposite.

Finally, as C++ was clearly not adapted to the researchers' needs of interactivity, 
Python bindings have also been provided for the standard mode of Essentia.
Python bindings for the streaming mode, being more complex to implement, were
dropped in favor of a specially designed `Domain-specific language`_ for defining
networks of algorithms, dubbed ESX (ESsentia eXtractor).

.. _Domain-specific language: http://en.wikipedia.org/wiki/Domain-specific_language


Essentia 1.0.x
^^^^^^^^^^^^^^

During the 1.0.x cycle, we performed a complete white box review of all the
algorithms available at that time, to ensure that APIs and naming were consistent,
that all algorithms had proper documentation (including scientific references),
and that they were also performing as intended, which lead to the writing of ~1000 unit tests. 
This huge review cycle ended with the 1.0.6 version, released in March 2009.


Essentia 1.1
^^^^^^^^^^^^

*(released August 2009)*

Python bindings for the streaming mode that were implemented as ESX resulted to be a maintenance
burden in the long-run and too limiting in the types of the possibilities it offered, and they
were dropped. Instead, full python bindings for the streaming mode have been written.

This release also introduced the new ffmpeg audio I/O, which allowed
to get audio from pretty much any source and in any format (even works
directly with youtube videos!)

More mid-level and high-level descriptors have also been added to complete the
list of available algorithms.


Essentia 1.2
^^^^^^^^^^^^

*(released April 2010)*

An algorithm that allows to apply Gaia transforms has been added. It allowed
to train classification models (SVM, nearest-neighbor, ...) and run them as
an Essentia feature extraction algorithm.

A more diverse set of prebuilt extractors has been written in order to provide
a useful out-of-the-box extractor suited for people more interested in machine learning
rather than feature extraction. The extractors included lowlevel features, rhythm, tonal/key, etc.

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

*(released October 2013)*

This major release is the first release to be publicly available as free
software.

It features a refactoring of the core API, a little bit for
the standard mode to fix small API annoyances, but mostly for the streaming
mode which is now much better defined, using sound computer science
techniques (The visible network is a `directed acyclic graph`_, the composites
have better defined semantics, and the order of execution of the algorithms
is the `topological sort`_ of the `transitive reduction`_ of the visible network
after the composites have been expanded). In particular, the scheduler that
runs the algorithms in the streaming mode
is now a lot more correct, which permitted to clean all the small hacks that
had accumulated in the algorithms themselves during the 1.x releases to
compensate for the deficiencies of the initial scheduler.

Furthermore, the 2.0 release features new state-of-the-art algorithms for onset detection, beat
tracking and melody extraction, new and updated features extractors, and an updated
version of the Essentia's Vamp plugin in addition to a number of bugfixes and thoroughly revised documentation.


.. _Algorithms overview: algorithms_overview.html
.. _Streaming architecture: streaming_architecture.html
.. _Python tutorial: python_tutorial.html
.. _"Standard" mode how-to: howto_standard_extractor.html
.. _"Streaming" mode how-to: howto_streaming_extractor.html
.. _directed acyclic graph: https://en.wikipedia.org/wiki/Directed_acyclic_graph
.. _topological sort: http://en.wikipedia.org/wiki/Topological_sorting
.. _transitive reduction: https://en.wikipedia.org/wiki/Transitive_reduction
