/**
 * @mainpage
 *
 * @section Essentia_Doxygen_Documentation Essentia Doxygen Documentation
 *
 * This page is the main entry point to the Essentia's Doxygen documentation.
 * This is intended to be a reference for the advanced programmer. For a more user-oriented
 * documentation, please refer to <a href="../index.html" target="_parent">Essentia's main documentation</a>.
 *
 * Although you have access to all the classes from here (through the navigation bar located
 * at the top of this page), You will probably be more interested in the following pages:
 *
 * @subsection Algorithms Algorithm-related classes
 *
 * essentia::Configurable is the base class that both essentia::standard::Algorithm and
 * essentia::streaming::Algorithm inherit.
 *
 * essentia::standard::Algorithm is the base class for algorithms written to be used
 * in the standard mode.
 *
 * essentia::streaming::Algorithm is the base class for algorithms written in the
 * streaming mode. Most of the time you will not want to use this class directly but
 * one of its subclasses:
 *
 *  - essentia::streaming::AlgorithmComposite for composite algorithms.
 *  - essentia::streaming::AccumulatorAlgorithm for algorithms that consume the whole
 *    stream of audio and produce a single value at the end.
 *  - essentia::streaming::StreamingAlgorithmWrapper for algorithms that are frame-based
 *    and that can easily wrap their couterpart standard algorithm.
 *
 * @subsection Connectors The input and output classes (connectors)
 *
 * In standard mode, you have the essentia::standard::Input and essentia::standard::Output classes.
 *
 * In streaming mode, you have the essentia::streaming::Sink and essentia::streaming::Source classes.
 *
 * @subsection Miscellaneous Miscellaneous
 *
 * essentia::Parameter is the class you want to use with essentia::Configurable objects to
 * configure them. They are most of the time contained in an essentia::ParameterMap.
 *
 * essentia::Pool is a class used to store values in a thread-safe way.
 */
