/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 */

#ifndef ESSENTIA_SCHEDULER_NETWORK_H
#define ESSENTIA_SCHEDULER_NETWORK_H

#include <vector>
#include <set>
#include <stack>
#include "../streaming/streamingalgorithm.h"
#include "../essentiautil.h"

namespace essentia {
namespace streaming {

class AlgorithmComposite;
class SourceBase;

} // namespace streaming
} // namespace essentia




namespace essentia {
namespace scheduler {

typedef std::vector<streaming::Algorithm*> AlgoVector;
typedef std::set<streaming::Algorithm*> AlgoSet;



/**
 * A NetworkNode is a structure used to represent an Algorithm in a Network
 * of Execution. It points to a given Algorithm and also contains a list of
 * algorithms which execution should come after this one's, hence forming the
 * basis for a tree structure.
 * TODO: rename dependencies to children?
 */
class NetworkNode {
 public:
  NetworkNode(streaming::Algorithm* algo) : _algo(algo) {}

  const std::vector<NetworkNode*>& children() const { return _children; }
  void setChildren(const std::vector<NetworkNode*>& children) { _children = children; }
  void addChild(NetworkNode* child) { if (!contains(_children, child)) _children.push_back(child); }

  const streaming::Algorithm* algorithm() const { return _algo; }
        streaming::Algorithm* algorithm()       { return _algo; }

  std::vector<NetworkNode*> addVisibleDependencies(std::map<streaming::Algorithm*, NetworkNode*>& algoNodeMap);

 protected:
  /**
   * Algorithm that this node represents in the network.
   */
  streaming::Algorithm* _algo;
  std::vector<NetworkNode*> _children;
};

typedef std::vector<NetworkNode*> NodeVector;
typedef std::set<NetworkNode*> NodeSet;
typedef std::stack<NetworkNode*> NodeStack;



/**
 * A Network is a structure that holds all algorithms that have been connected
 * together and is able to run them.
 *
 * It contains 2 networks of algorithms:
 *  - the visible network, which is the network of algorithms explicitly
 *    connected together by the user, which can contain AlgorithmComposites
 *  - the execution network, where all the AlgorithmComposites have been
 *    replaced with their respective constituent algorithms, so that only
 *    non-composite algorithms are left.
 *
 * The main functionality of the Network, once it is built, is to run the
 * generator node at the root of the Network (an audio loader, usually) and
 * carry the data through all the other algorithms automatically.
 */
class Network {

 public:
  /**
   * Builds an execution Network using the given Algorithm.
   * This will only work if the given algorithm is a generator, ie: it has no
   * input ports.
   * @param generator the root generator node to which all the network is connected
   * @param takeOwnership whether to take ownership of the algorithms and delete
   *        them when this Network object is destroyed
   */
  Network(streaming::Algorithm* generator, bool takeOwnership = true);

  ~Network();

  void run();

  /**
   * Rebuilds the visible and execution network.
   */
  void update() {
    buildVisibleNetwork();
    buildExecutionNetwork();
    topologicalSortExecutionNetwork();
  }

  /**
   * Reset all the algorithms contained in this network.
   * (This in effect calls their reset() method)
   */
  void reset();

  /**
   * Clear the network to an empty state (ie: no algorithms contained in it,
   * delete previous algorithms if Network had ownership of them).
   * This is the recommended way to proceed (do not use deleteAlgorithms()
   * unless you really know what you're doing).
   */
  void clear();

  /**
   * Delete all the algorithms contained in this network.
   * Be careful as this method will indeed delete all the algorithms in the
   * Network, even if it didn't take ownership over them.
   */
  void deleteAlgorithms();


  /**
   * Find and return an algorithm by its name.
   * Throw an exception if no algorithm was found with the given name.
   */
  streaming::Algorithm* findAlgorithm(const std::string& name);

  NetworkNode* visibleNetworkRoot() { return _visibleNetworkRoot; }
  NetworkNode* executionNetworkRoot() {
    if (_visibleNetworkRoot && !_executionNetworkRoot) {
      buildExecutionNetwork();
    }
    return _executionNetworkRoot;
  }

  /**
   * Return a list of algorithms which have been topologically sorted.
   * You can assume that each node in there will have either 1 or 0 children.
   */
  const std::vector<streaming::Algorithm*>& linearExecutionOrder() const { return _toposortedNetwork; }


  /**
   * Helper function that returns the list of visibly connected algorithms
   * starting from the given one, without crossing the borders of a possibly
   * encompassing AlgorithmComposite (ie: all returned algorithms are inside
   * the composite).
   */
  static std::vector<streaming::Algorithm*> innerVisibleAlgorithms(streaming::Algorithm* algo);

  /**
   * Prints the fill state of all the buffers in the network.
   */
  void printBufferFillState();

  /**
   * Last instance of Network created, 0 if it has been deleted or if
   * no network has been created yet.
   */
  static Network* lastCreated;

 protected:
  bool _takeOwnership;
  streaming::Algorithm* _generator;
  NetworkNode* _visibleNetworkRoot;
  NetworkNode* _executionNetworkRoot;
  std::vector<streaming::Algorithm*> _toposortedNetwork;

  /**
   * Build the network of visibly connected algorithms (ie: do not enter composite
   * algorithms) and stores its root in @c _visibleNetworkRoot.
   */
  void buildVisibleNetwork();

  /**
   * Build the network execution, ie: the network of single algorithms in the order
   * in which they should be executed. All composite algorithms should have been expanded
   * and the only remaining composites in there should be those that call themselves as
   * part of a @c declareProcessStep() call in the declareProcessOrder() method.
   */
  void buildExecutionNetwork();

  /**
   * Perform a topological sort of the execution network and store it internally.
   */
  void topologicalSortExecutionNetwork();

  /**
   * Execution dependencies are stored inside the network nodes themselves, and
   * might enter/exit CompositeAlgorithms boundaries.
   * This variable does not do that and only keeps the simple list of algorithms,
   * for the purpose of resetting/deleting them.
   */
  // TODO: rename AlgoSet visibleAlgos() const;
  AlgoSet _algos;

  /**
   * Check that all the algorithms inputs/outputs are connected somewhere, so as
   * to make sure that no buffer is being filled without anybody to empty it,
   * which would cause the whole network to be blocked.
   * This function returns normally if it didn't find any problems, otherwise it
   * throws an exception.
   */
  void checkConnections();

  /**
   * Check for all the connections that the source buffer size (phantom size,
   * actually) is at least as big as the preferred size of the connected sink.
   * If not, it automatically resizes the source buffer.
   */
  void checkBufferSizes();

  /**
   * Delete all the NetworkNodes used in the visible network. Do not touch the
   * algorithms pointed to by these nodes.
   */
  void clearVisibleNetwork();

  /**
   * Delete all the NetworkNodes used in the execution network. Do not touch the
   * algorithms pointed to by these nodes.
   */
  void clearExecutionNetwork();
};

/**
 * Prints the fill state of all the buffers in the last created network.
 */
void printNetworkBufferFillState();

AlgoVector computeDependencies(const streaming::Algorithm* algo);
AlgoVector computeNormalDependencies(const streaming::Algorithm* algo);
AlgoVector computeCompositeDependencies(const streaming::Algorithm* algo);

/**
 * This function computes at once all the dependencies for the network starting
 * at the given algorithm and caches each algorithm dependencies inside the algo
 * instance. This means that we don't have to do this each time in the scheduler,
 * and also avoid doing any allocation while running the network.
 */
void cacheDependencies(streaming::Algorithm* algo);


/**
 * Returns the list of all algorithms which live inside of the given composite
 * algorithm.
 */
AlgoSet compositeInnerAlgos(streaming::Algorithm* algo);

/**
 * Returns the set of algorithms which are in the branch starting from the given
 * one and going up from parent to parent. It stops whenever it goes outside of
 * the specified composite algorithm.
 */
AlgoSet parentBranchInsideComposite(streaming::AlgorithmComposite* composite,
                                    streaming::Algorithm* algo);

} // namespace scheduler
} // namespace essentia


#endif // ESSENTIA_SCHEDULER_NETWORK_H
