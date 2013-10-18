
Execution network algorithm
===========================


Problem description
-------------------

This algorithm describes how to get the execution network from a node after fully expanding its
composite algorithms.

The requirements are:

* a network of connected algorithms (composite or not). They are connected via their output sources
  which connect to one or more input sink(s). Every sink is only connected to a single source.
  Composites can only have SourceProxy/SinkProxy.

* Composite define a ``declareProcessOrder()`` method, which defines the steps to be taken when
  this algorithm should be executed. They can be one of two types:

  * ``ChainFrom(algo)``: which runs the given algorithm and all its dependencies which are contained
    inside the Composite
  * ``SingleShot(algo)``: which runs the given algorithm once

  As the steps are declared in sequential order in the ``declareProcessOrder()`` method, they should
  also be run that way, which means that the next step should depend on the completion of all the
  previous ones [1]_.

Composite algorithms are described more in details in the :doc:`composite_api` page.


Algorithm description
---------------------

The following algorithms is used. It is written in (weird) pseudo-code that is a mix of C++
(for showing types) and Python (for avoiding boilerplate code that comes with C++). Hopefully,
it makes for a clear reading.


Structures used
^^^^^^^^^^^^^^^

.. highlight:: c++

::

    // A Node represents a node in the execution graph, and as such points to an algorithm it
    // represents and has a list of children which execution should come after this one.
    class Node {

        // the algorithm that this node represents in the execution tree
        Algorithm* algo;

        // the algorithms that need to be run after this one has completed its execution
        vector<Node*> children;

    };


::

    // A FractalNode represents a node that can be expanded, by recursively replacing a Composite
    // algorithm with its constituent parts. It is a temporary structure only used while computing
    // the execution network from the visible network.
    class FractalNode : public Node {

        // the expanded version of this fractal node
        FractalNode* expanded;

        // for each source name, the list of expanded (ie: not composite) algorithm which execution
        // should be completed before this source is allowed to produce data
        map<string, vector<FractalNode*> > innerMap;

        // for each source name, the list of visible (ie: composite or not) connected algorithms,
        // that is, the list of algorithms that need to be run after this source has produced data
        map<string, vector<FractalNode*> > outputMap;

    };



Detailed algorithm
^^^^^^^^^^^^^^^^^^

The algorithm we will use can be summarily described as:

1. Build visible network from root node using FractalNodes. The visible network is the network we
   obtain when looking at the explicit connections made by the user.

2. Build another graph with the same topology where all the nodes have been expanded (i.e.: the
   Composite algorithms have been replaced by their constituent parts).

3. Using the connections defined in the first graph, reconnect all nodes in the second graph.
   This is not as trivial as it seems as source/sink proxies might have different names than the
   connector they relay.


The pseudo-code algorithm is the following:

.. highlight:: python

::

    def buildExecutionNetwork(rootAlgorithm):

        # 1- build visible network: this is the tree obtained by setting as children of a node N
        #    all algorithms which have a sink connected to a source of the algorithm pointed to
        #    by node N
        FractalNode* executionNetworkRoot = visibleNetwork(rootAlgorithm)

        # 2- expand all nodes of this first graph
        for node in DFS(executionNetworkRoot):
            node.expanded = expandNode(node)

        # 3- connect expanded network
        for node in DFS(executionNetworkRoot):

            # connectedNodes is the list of externally connected nodes to the given output
            for outputName, connectedNodes in node.outputMap:

                # innerNodes (= node.expanded.innerMap[outputName]) is the list of nodes
                # from inside the composite connected to a given output. Although we can only
                # have 1 source connected to a SourceProxy, we can have multiple algorithms
                # that we have to wait for before computing the next algorithm in the tree.
                for innerNode in node.expanded.innerMap[outputName]:

                    # for each expanded node inside the algorithm which outputs data on a given
                    # source (output), and for each connected algorithm on this source...
                    for cnode in connectedNodes:

                        # ... we add the expanded version of the connected algorithm as a
                        # dependency for the inner node.
                        connect(innerNode,       # expanded algo inside the composite
                                cnode.expanded)  # expanded algo outside, i.e.: inside the visible dependency


        # 4- clean up our temporary structure and return the execution network
        return cleanedUpExecutionNetworkRoot



    # This function expands a given node and fills its innerMap during the process
    def expandNode(node):
        if not is_composite(node):
            # non-composite algorithm: all the inner connections are on the algorithm itself
            for outputName in node.algorithm.outputNames:
                node.expanded.innerMap[outputName] = [ node.expanded ]

        else:
            # node is a composite algorithm
            for step in algo.processOrder:
                stepRoot = step.algorithm

                if step.type == 'single':
                    fillInnerMapWithConnections(stepRoot)
                    stepRoot.expanded = expandNode(stepRoot)

                elif step.type == 'chain':
                    # simplified; should also fill stepRoot.innerMap while doing this
                    stepRoot.expanded = buildExecutionNetwork(stepRoot)





At the end, we should obtain a `Hasse Diagram <http://en.wikipedia.org/wiki/Hasse_diagram>`_
as a result.


Execution of the network
------------------------

There are 2 main ways of running a network:

- the *single-threaded* way: in this case, we need to
  `topologically sort <http://en.wikipedia.org/wiki/Topological_sorting>`_  the network in order to
  get the execution order. Once we have the topological order, we can run each algorithm sequentially
  until the generator signals us that it is over.

- the *multi-threaded* way: in this case, we will have to create tasks (using a wavefront
  pattern [2]_, for instance) for a task library, such as Intel TBB, and let its scheduler run them.
  Intel TBB's scheduler seems very adequate, in the sense that it does it in a computer friendly way:
  any topological ordering is mathematically correct, but it uses
  `DFS <http://en.wikipedia.org/wiki/Depth-first_search>`_ to build it so as to allow better cache
  use, and steals tasks from the top of the queue [3]_.



.. [1] this may lead to situations where we create lots of unnecessary dependencies in the graph.
       This is not a problem as we can reduce it thereafter with a
       `transitive reduction <http://en.wikipedia.org/wiki/Transitive_reduction>`_.
.. [2] as shown in pattern 5 of `Intel TBB's design patterns <http://www.threadingbuildingblocks.org/uploads/81/91/Latest%20Open%20Source%20Documentation/Design_Patterns.pdf>`_.
.. [3] For more information about TBB's scheduler, please refer to
       `TBB reference manual <http://www.threadingbuildingblocks.org/uploads/81/91/Latest%20Open%20Source%20Documentation/Reference.pdf>`_
       or to some of the blogs aggregated `here <http://software.intel.com/en-us/blogs/category/intel-threading-building-blocks/>`_.
