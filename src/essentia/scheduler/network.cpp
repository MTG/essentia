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

#include <stack>
#include "network.h"
#include "graphutils.h"
#include "../streaming/streamingalgorithm.h"
#include "../streaming/streamingalgorithmcomposite.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;

namespace essentia {
namespace scheduler {

// helper function, was inside Algorithm before but it makes more sense to have
// it only here, statically defined (ie: not part of Algorithm API)
set<Algorithm*> visibleDependencies(const Algorithm* algo) {
  set<Algorithm*> dependencies;

  // for each source of this algorithm...
  for (Algorithm::OutputMap::const_iterator output = algo->outputs().begin();
       output != algo->outputs().end();
       ++output) {

    // we always want to stop whenever we go out of a CompositeAlgorithm, ie:
    // whenever 1 of our outputs is actually proxied
    if (output->second->isProxied()) continue;

    vector<SinkBase*>& sinks = output->second->sinks();

    // ...get the attached sinks and their parent algorithms
    for (vector<SinkBase*>::iterator it = sinks.begin(); it != sinks.end(); ++it) {
      // add the owning algorithm to the list of dependencies
      dependencies.insert((*it)->parent());
    }
  }

  return dependencies;
}

/**
 * Return the map of <source_name, connected_algo*> for all sources which are
 * not proxies (ie: all connections that do not cross the boundary of a
 * composite algorithm)
 */
map<string, vector<Algorithm*> > mapVisibleDependencies(const Algorithm* algo) {
  map<string, vector<Algorithm*> > result;

  // for each source of this algorithm...
  for (Algorithm::OutputMap::const_iterator output = algo->outputs().begin();
       output != algo->outputs().end();
       ++output) {

    vector<SinkBase*>& sinks = output->second->sinks();

    // ...get the attached sinks and their parent algorithms
    for (vector<SinkBase*>::iterator it = sinks.begin(); it != sinks.end(); ++it) {
      // if the sink is connected through the proxy, don't follow it
      if (output->second->isProxied() &&
          indexOf(output->second->proxiedSinks(), *it) != -1) continue;

      // otherwise add the owning algorithm to the list of dependencies
      result[output->second->name()].push_back((*it)->parent());
    }
  }

  return result;
}



template <typename NodeType>
vector<NodeType*> nodeDependencies(const Algorithm* algo) {
  vector<NodeType> result;
  set<Algorithm*> dependencies = visibleDependencies(algo);

  for (set<Algorithm*>::iterator it = dependencies.begin(); it != dependencies.end(); ++it) {
    result.push_back(new NodeType(*it));
  }

  return result;
}

template <typename NodeType>
NodeType* visibleNetwork(Algorithm* algo) {
  stack<NodeType*> toVisit;
  set<NodeType*> visited;
  map<Algorithm*, NodeType*> algoNodeMap;

  NodeType* networkRoot = new NodeType(algo);
  toVisit.push(networkRoot);

  E_DEBUG(ENetwork, "building visible network from " << algo->name());
  E_DEBUG_INDENT;

  while (!toVisit.empty()) {
    NodeType* currentNode = toVisit.top();
    toVisit.pop();
    //E_DEBUG(ENetwork, "visiting: " << currentNode->algorithm()->name());

    if (visited.find(currentNode) != visited.end()) continue;
    visited.insert(currentNode);

    vector<NodeType*> deps = currentNode->addVisibleDependencies(algoNodeMap);

    E_DEBUG(ENetwork, currentNode->algorithm()->name() << ":");
    for (int i=0; i<(int)deps.size(); i++) {
      E_DEBUG(ENetwork, "  → " << deps[i]->algorithm()->name());
      toVisit.push(deps[i]);
    }
  }

  E_DEBUG_OUTDENT;
  E_DEBUG(ENetwork, "building visible network from " << algo->name() << " ok!");
  return networkRoot;
}


vector<NetworkNode*> NetworkNode::addVisibleDependencies(map<Algorithm*, NetworkNode*>& algoNodeMap) {
  set<Algorithm*> dependencies = visibleDependencies(_algo);
  vector<NetworkNode*> nodeChildren;

  for (set<Algorithm*>::iterator it = dependencies.begin(); it != dependencies.end(); ++it) {
    if (!contains(algoNodeMap, *it)) {
      algoNodeMap[*it] = new NetworkNode(*it);
    }

    nodeChildren.push_back(algoNodeMap[*it]);
  }

  setChildren(nodeChildren);

  return nodeChildren;
}


Network* Network::lastCreated = 0;

Network::Network(Algorithm* generator, bool takeOwnership) : _takeOwnership(takeOwnership),
                                                             _generator(generator),
                                                             _visibleNetworkRoot(0),
                                                             _executionNetworkRoot(0) {
  lastCreated = this;

  // 1- find the simple list of algorithms connected in this network
  buildVisibleNetwork();

  // 2- break up the CompositeAlgorithms boundaries and create the execution network
  // Note: this is done in the run() method, just before it is actually needed, as we might
  //       reconfigure a network and then have one of its composite change its internal
  //       structure (ie: add/remove resampling, ...)
}

Network::~Network() {
  if (lastCreated == this) lastCreated = 0;
  clear();
}

void Network::clear() {
  if (_takeOwnership) {
    deleteAlgorithms();
  }

  // delete all our networks
  clearVisibleNetwork();
  clearExecutionNetwork();
}

void Network::clearVisibleNetwork() {
  E_DEBUG(ENetwork, "Network::clearVisibleNetwork()");
  vector<NetworkNode*> nodes = depthFirstSearch(_visibleNetworkRoot);
  for (int i=0; i<(int)nodes.size(); i++) delete nodes[i];
  _visibleNetworkRoot = 0;
  E_DEBUG(ENetwork, "Network::clearVisibleNetwork() ok!");
}

void Network::clearExecutionNetwork() {
  E_DEBUG(ENetwork, "Network::clearExecutionNetwork()");
  vector<NetworkNode*> nodes = depthFirstSearch(_executionNetworkRoot);
  for (int i=0; i<(int)nodes.size(); i++) delete nodes[i];
  _executionNetworkRoot = 0;
  E_DEBUG(ENetwork, "Network::clearExecutionNetwork() ok!");
}


/**
 * Small helper function that returns whether an algorithm has produced any tokens
 * since the last time this function was called.
 * FIXME: deprecate (?)
 */
bool algorithmHasProduced(Algorithm* algo, EssentiaMap<SourceBase*, int>& produced) {
  bool hasProduced = false;
  for (int i=0; i<(int)algo->outputs().size(); i++) {
    SourceBase* output = &algo->output(i);
    int before = produced[output];
    int now = output->totalProduced();
    if (now > before) {
      hasProduced = true;
      produced[output] = now;
    }
  }
  return hasProduced;
}

void Network::run() {
  // 1- build the execution network here as internal configuration of some
  //    algorithms might have changed since we constructed the Network
  buildExecutionNetwork();

  // 2- get a linear ordering on the newly constructed execution network
  topologicalSortExecutionNetwork();

  // 3- make sure all inputs/outputs are correctly connected
  checkConnections();

  // 4- resize the buffers depending on the requirements of the connected sinks
  checkBufferSizes();

  // 5- actually run the network
  if (_toposortedNetwork.empty()) return;

  // keep a map of total tokens produced by each output of each algorithm, that
  // way we can know whether an algorithm produced data or not and re-run it if
  // necessary
  /*
  EssentiaMap<SourceBase*, int> produced;
  for (int i=0; i<(int)_toposortedNetwork.size(); i++) {
    Algorithm* algo = _toposortedNetwork[i];
    for (int j=0; j<(int)algo->outputs().size(); j++) {
      produced.insert(&algo->output(j), 0);
    }
  }
  */

#if DEBUGGING_ENABLED
  for (int i=0; i<(int)_toposortedNetwork.size(); i++) _toposortedNetwork[i]->nProcess = 0;
#endif

  streaming::Algorithm* gen = _toposortedNetwork[0];
  bool endOfStream = false;
  string dash(24, '-');

  saveDebugLevels();

  while (!gen->shouldStop()) {
    // first run the generator once
#if DEBUGGING_ENABLED
    restoreDebugLevels();
    setDebugLevelForTimeIndex(gen->nProcess);
    E_DEBUG(ENetwork, "-------- Running generator loop index " << gen->nProcess << " --------");

    E_DEBUG(EScheduler, dash << " Buffer states before running generator, nProcess = " << gen->nProcess << " " << dash);
    printNetworkBufferFillState();
#endif
    gen->process();

    endOfStream = gen->shouldStop();

#if DEBUGGING_ENABLED
    gen->nProcess++;

    if (endOfStream) E_DEBUG(ENetwork, "Generator " << gen->name() << " run " <<
                             gen->nProcess << " times, shouldStop = true " <<
                             "(end of stream reached, and all tokens produced)");

    //E_DEBUG(EScheduler, dash << " Buffer states after running generator " << dash);
    //printBufferFillState();
#endif

    // then run each algorithm as many times as needed for them to consume everything on their input
    stack<int> runStack;
    runStack.push(1);
    while (!runStack.empty()) {
      int startIndex = runStack.top();
      runStack.pop();

      for (int i=startIndex; i<(int)_toposortedNetwork.size(); i++) {
        // only propagate the end of stream marker as long as we don't have any
        // algorithm rescheduled to run
        _toposortedNetwork[i]->shouldStop(endOfStream && runStack.empty());
        AlgorithmStatus status;
        do {
          status = _toposortedNetwork[i]->process();

#if DEBUGGING_ENABLED
          if (status == OK || status == FINISHED) _toposortedNetwork[i]->nProcess++;
#endif

          // if status == NO_OUTPUT, push i on a stack to remember to execute it again later;
          // execute all of its dependencies, then pop i from the stack and reexecute, as well
          // as the dependencies
          // NOTE: be careful with endOfStream, it should not be propagated
          // as long as we have at least 1 index value on the stack
          if (status == NO_OUTPUT) {
            runStack.push(i);
            E_DEBUG(EScheduler, "Rescheduling algorithm " << _toposortedNetwork[i]->name() <<
                    " on generator frame " << gen->nProcess <<
                    " to run later, output buffers temporarily full");
            /*
            E_WARNING("Rescheduling algorithm " << _toposortedNetwork[i]->name() <<
                      " on generator frame " << gen->nProcess <<
                      " to run later, output buffers temporarily full");
            E_WARNING("You may want to consider resizing one of the output buffers of " <<
                      "this algorithm for better performance");
            */
            printNetworkBufferFillState();
          }
        } while (status == OK);

      }
    }
    E_DEBUG(EScheduler, dash << " Buffer states after running the generator and all the nodes " << dash);
    printBufferFillState();
  }

  E_DEBUG(ENetwork, dash << " Final buffer states " << dash);
  printBufferFillState();
}

Algorithm* Network::findAlgorithm(const std::string& name) {
  NodeVector nodes = depthFirstSearch(_visibleNetworkRoot);
  for (NodeVector::iterator node = nodes.begin(); node != nodes.end(); ++node) {
    if ((*node)->algorithm()->name() == name) return (*node)->algorithm();
  }

  ostringstream msg;
  msg << "Could not find algorithm with name '" << name << "'. Known algorithms are: ";
  if (!nodes.empty()) msg << '\'' << nodes[0]->algorithm()->name() << '\'';
  for (int i=1; i<(int)nodes.size(); i++) {
    msg << ", '" << nodes[i]->algorithm()->name() << '\'';
  }
  throw EssentiaException(msg);
}


void Network::reset() {
  NodeVector nodes = depthFirstSearch(_visibleNetworkRoot);
  for (NodeVector::iterator node = nodes.begin(); node != nodes.end(); ++node) {
    (*node)->algorithm()->reset();
  }
}

void Network::deleteAlgorithms() {
  E_DEBUG(ENetwork, "Network::deleteAlgorithms()");

  NodeVector nodes = depthFirstSearch(_visibleNetworkRoot);
  for (NodeVector::iterator node = nodes.begin(); node != nodes.end(); ++node) {
    E_DEBUG(ENetwork, "deleting " << (*node)->algorithm()->name());
    delete (*node)->algorithm();
  }

  // we need to set this to false anyway, because it doesn't make sense anymore
  // to have it to true and it would cause the destructor to crash
  _takeOwnership = false;

  E_DEBUG(ENetwork, "Network::deleteAlgorithms() ok!");
}


void Network::buildVisibleNetwork() {
  clearVisibleNetwork();
  E_DEBUG(ENetwork, "Network::buildVisibleNetwork()");
  _visibleNetworkRoot = visibleNetwork<NetworkNode>(_generator);
}


vector<Algorithm*> Network::innerVisibleAlgorithms(Algorithm* algo) {
  NetworkNode* visibleNetworkRoot = visibleNetwork<NetworkNode>(algo);

  vector<Algorithm*> algos = depthFirstMap(visibleNetworkRoot, returnAlgorithm);

  NodeVector nodes = depthFirstSearch(visibleNetworkRoot);
  for (int i=0; i<(int)nodes.size(); i++) delete nodes[i];

  return algos;
}



class FractalNode : public NetworkNode {
 public:
  // the expanded version of this fractal node
  FractalNode* expanded;

  typedef map<string, vector<FractalNode*> > NodeMap;

  // for each source name, the list of expanded (ie: not composite) algorithm which execution
  // should be completed before this source is allowed to produce data
  NodeMap innerMap;

  // for each source name, the list of visible (ie: composite or not) connected algorithms,
  // that is, the list of algorithms that need to be run after this source has produced data
  NodeMap outputMap;


 public:
  FractalNode(Algorithm* algo) : NetworkNode(algo) {}

  const vector<FractalNode*>& children() const { return _fchildren; }
        vector<FractalNode*>& children()       { return _fchildren; }

  void addChild(FractalNode* child) { if (!essentia::contains(_fchildren, child)) _fchildren.push_back(child); }

  // we need to overload this method, because in the case of FractalNodes, we want
  // to fill the OutputMap at the same time with the names of the outputs
  vector<FractalNode*> addVisibleDependencies(map<Algorithm*, FractalNode*>& algoNodeMap) {
    E_DEBUG(ENetwork, "add visible deps to " << _algo->name());
    map<string, vector<Algorithm*> > namedDeps = mapVisibleDependencies(_algo);
    //E_DEBUG(ENetwork, "name deps size: " << namedDeps.size());

    for (map<string, vector<Algorithm*> >::iterator connection = namedDeps.begin();
         connection != namedDeps.end();
         ++connection) {
      const string& outputName = connection->first;
      //E_DEBUG(ENetwork, "--" << outputName);
      const vector<Algorithm*>& connectedAlgos = connection->second;

      for (int i=0; i<(int)connectedAlgos.size(); i++) {
        if (!contains(algoNodeMap, connectedAlgos[i])) {
          algoNodeMap[connectedAlgos[i]] = new FractalNode(connectedAlgos[i]);
          _fchildren.push_back(algoNodeMap[connectedAlgos[i]]);
        }

        this->outputMap[outputName].push_back(algoNodeMap[connectedAlgos[i]]);
      }
    }

    return _fchildren;
  }

 protected:
  vector<FractalNode*> _fchildren;
};


typedef std::vector<FractalNode*> FNodeVector;
typedef std::set<FractalNode*> FNodeSet;
typedef std::stack<FractalNode*> FNodeStack;

FractalNode* expandNode(FractalNode* node);

void printInnerMap(const map<string, vector<FractalNode*> >& innerMap) {
  for (map<string, vector<FractalNode*> >::const_iterator it = innerMap.begin(); it != innerMap.end(); ++it) {
    for (int i=0; i<(int)it->second.size(); i++) {
      E_DEBUG(ENetwork, "output " <<  it->first << " → " << it->second[i]->algorithm()->name());
    }
  }
}

void expandNodes(FNodeVector& visibleNodes) {
  E_DEBUG(ENetwork, "visible nodes:" << visibleNodes.size());

  for (int i=0; i<(int)visibleNodes.size(); i++) {
    E_DEBUG(ENetwork, "expanding " << visibleNodes[i]->algorithm()->name());
    visibleNodes[i]->expanded = expandNode(visibleNodes[i]);
    E_DEBUG(ENetwork, "expanded " << visibleNodes[i]->algorithm()->name() <<
            " to " << visibleNodes[i]->expanded->algorithm()->name());
  }
}

void connectExpandedNodes(FNodeVector& visibleNodes) {
  for (int i=0; i<(int)visibleNodes.size(); i++) {
    FractalNode* node = visibleNodes[i];
    E_DEBUG(ENetwork, "    node: " << node->algorithm()->name() << " - " << node->outputMap.size() << " outputs");

    for (FractalNode::NodeMap::iterator output = node->outputMap.begin();
         output != node->outputMap.end(); ++output) {
      const string& outputName = output->first;
      const vector<FractalNode*>& dnodes = output->second;
      //E_DEBUG(ENetwork, "        output: " << outputName << " - " << dnodes.size() << " connected algos");

      for (int j=0; j<(int)dnodes.size(); j++) {
        vector<FractalNode*>& lnodes = node->expanded->innerMap[outputName]; // algos inside the composite
        for (int k=0; k<(int)lnodes.size(); k++) {
          FractalNode* lhs = lnodes[k];
          FractalNode* rhs = dnodes[j]->expanded;                  // algo outside, ie: inside the visible dependency
          lhs->addChild(rhs);
          E_DEBUG(ENetwork, "            actual: " << lhs->algorithm()->name() << "::" << outputName << " → " << rhs->algorithm()->name());
        }
      }
    }
  }
}

/**
 * Class that maintains a list of declared SourceProxies for an AlgorithmComposite and
 * that is able to track scheduled algorithms which are attached to those proxies while
 * the execution network is being built.
 */
class ProxyMatcher {
 protected:
  Algorithm* _algo; // the composite algorithm for which we're matching the proxy sources

  // map of (external (proxy) source name → <inner algo, inner source name>)
  map<string, pair<Algorithm*, string> > _proxiedSources;

  // map of algorithm to the last visited FractalNode pointing to it
  map<Algorithm*, FractalNode*> _lastVisited;

 public:
  ProxyMatcher(AlgorithmComposite* algo) : _algo(algo) {
    for (int i=0; i<(int)algo->outputs().size(); i++) {
      SourceBase* sbase = &algo->output(i);
      SourceProxyBase* sproxy = dynamic_cast<SourceProxyBase*>(sbase);

      if (sproxy) {
        Algorithm* innerAlgo = sproxy->proxiedSource()->parent();
        const string& innerSourceName = sproxy->proxiedSource()->name();
        const string& proxyName = sproxy->name();

        _proxiedSources[proxyName] = make_pair(innerAlgo, innerSourceName);
        _lastVisited[innerAlgo] = 0;
      }
      else {
        // if it was not a proxy, then the Source belongs directly to the composite and it
        // needs to be scheduled by itself. In any case, this can work the same as proxy,
        // just pretend the algorithm uses one.
        Algorithm* innerAlgo = algo;
        const string& innerSourceName = sbase->name();
        const string& proxyName = sbase->name();

        _proxiedSources[proxyName] = make_pair(innerAlgo, innerSourceName);
        _lastVisited[innerAlgo] = 0;
      }
    }
  }

  void clear() {
    for (map<Algorithm*, FractalNode*>::iterator it = _lastVisited.begin(); it != _lastVisited.end(); ++it) {
      it->second = 0;
    }
  }

  /**
   * Visit a (visible) FractalNode, and if the algo it points to is connected to one of our
   * composite's proxies, then remember it.
   * Return true if the algo is connected to a proxy, false otherwise.
   */
  bool visit(FractalNode* node) {
    if (contains(_lastVisited, node->algorithm())) {
      _lastVisited[node->algorithm()] = node;
      return true;
    }
    return false;
  }

  void printMatches() {
    E_DEBUG(ENetwork, "******************************");
    E_DEBUG(ENetwork, "Visible connections:");
    for (map<string, pair<Algorithm*, string> >::iterator it = _proxiedSources.begin(); it != _proxiedSources.end(); ++it) {
      E_DEBUG(ENetwork, "output " << it->first << " → " << it->second.first->name() << "::" << it->second.second);
    }
    E_DEBUG(ENetwork, "******************************");
    E_DEBUG(ENetwork, "Actual dependency:");
    printInnerMap(proxyMap(vector<FractalNode*>()));
    E_DEBUG(ENetwork, "******************************");
  }

  map<string, vector<FractalNode*> > proxyMap(const vector<FractalNode*>& previousLeaves) {
    // _proxyMap contains pointers to the non-expanded versions of the algorithms,
    // we need to expand that now

    // first check whether we visited all connected algos; if we didn't we have a broken
    // scheduling policy
    // NB: this is no longer valid, because now _lastVisited only remembers the nodes visited during the last process step.
    /*
    for (map<Algorithm*, FractalNode*>::iterator it = _lastVisited.begin(); it != _lastVisited.end(); ++it) {
      if (!it->second) {
        ostringstream msg;
        msg << "Broken scheduling policy: algorithm '" << _algo->name()
            << "' has at least 1 SourceProxy connected to its inner algorithm '" << it->first->name()
            << "' but the latter has not been scheduled";
        throw EssentiaException(msg);
      }
    }
    */

    // now replace all source proxies by their corresponding inner algorithms dependencies
    map<string, vector<FractalNode*> > result;
    for (map<string, pair<Algorithm*, string> >::iterator it = _proxiedSources.begin(); it != _proxiedSources.end(); ++it) {
      const string& outputName = it->first;
      Algorithm* innerAlgo = it->second.first;
      const string& innerSourceName = it->second.second;
      if (_lastVisited[innerAlgo]) {
        // if we visited the algo since last step, then all is good
        result[outputName] = _lastVisited[innerAlgo]->expanded->innerMap[innerSourceName];
      }
      else {
        // else we need to depend on the leaves of the last step to respect the chronological order
        result[outputName] = previousLeaves;
      }
    }

    return result;
  }
};

FractalNode* expandNonCompositeNode(FractalNode* node) {
  FractalNode* expanded = new FractalNode(node->algorithm());

  // standard algorithm: all the inner connections are on the algorithm itself
  vector<string> outputNames = node->algorithm()->outputNames();
  for (int i=0; i<(int)outputNames.size(); i++) {
    expanded->innerMap[outputNames[i]] = vector<FractalNode*>(1, expanded);
  }

  return expanded;
}

FractalNode* expandNode(FractalNode* node) {
  E_DEBUG_INDENT;

  AlgorithmComposite* calgo = dynamic_cast<AlgorithmComposite*>(node->algorithm());
  if (calgo) {
    // composite algorithm: we need to expand it
    vector<ProcessStep> processOrder = calgo->processOrder();

    if (processOrder.empty()) {
      throw EssentiaException("You forgot to specify a process order for the composite algorithm '", calgo->name(), "'");
    }

    // dummy root node for the chain, will be removed before returning the composite order
    FractalNode* froot = new FractalNode(0);

    // leaves of the previous step, so that the next one can depend on all of them
    // (in case a ChainFrom branches, we need to wait on the full execution of it, which
    // is equivalent to waiting on the leaves of it)
    vector<FractalNode*> previousLeaves(1, froot);

    // TODO: (maybe not here) check that all the ProxySinks are connected to the same
    //       algorithm, otherwise we're in trouble
    ProxyMatcher pmatch(calgo);

    vector<FractalNode*> sroots; // used for cleaning up afterwards

    // process each step one after the other, while attaching the root of the next one to
    // the leaves of the previous one
    for (int p=0; p<(int)processOrder.size(); p++) {
      ProcessStep& pstep = processOrder[p];
      FractalNode* stepRoot = new FractalNode(pstep.algorithm());
      sroots.push_back(stepRoot);
      pmatch.clear();

      if (pstep.algorithm() == calgo) {
        // if a composite algorithm is trying to schedule itself, we need to do some
        // special processing
        if (pstep.type() == "chain") {
          throw EssentiaException("You are trying to chain the composite algorithm '", calgo->name(),
                                  "' from within itself; this is forbidden. Use SingleShot in that case.");
        }
        else if (pstep.type() == "single") {
          // if a composite is scheduling itself, we shouldn't try to expand it but return
          // it just as if it were a non-composite, otherwise we go into an infinite loop
          E_DEBUG(ENetwork, "---------------------------------------------");
          E_DEBUG(ENetwork, "  SINGLE RECURSIVE SHOT FOR " << pstep.algorithm()->name());
          pmatch.visit(stepRoot);
          stepRoot->expanded = expandNonCompositeNode(stepRoot);
          E_DEBUG(ENetwork, "------------ SINGLE RECURSIVE SHOT DONE ---------------");
        }
        else throw EssentiaException("Unknown process order step: ", pstep.type());
      }
      else {
        // we are trying to schedule an algorithm which is not the composite itself, good! :-)

        if (pstep.type() == "chain") {
          // 1- build execution network starting from step->algo
          //      note all nodes which are attached to a proxy source/sink

          E_DEBUG(ENetwork, "-------------------------------------------------------------");
          E_DEBUG(ENetwork, "  Process step " << p << " - CHAIN FROM " << pstep.algorithm()->name());
          E_DEBUG(ENetwork, "  --1-- build visible network");
          FNodeStack toVisit;
          map<Algorithm*, FractalNode*> algoNodeMap;
          // FIXME: should check for already visited nodes here, same as for visible network
          toVisit.push(stepRoot);
          while (!toVisit.empty()) {
            FractalNode* currentNode = toVisit.top();
            toVisit.pop();

            E_DEBUG(ENetwork, "visiting " << currentNode->algorithm()->name());

            // if we're connected to a proxy, remember it and stop here
            // FIXME: wrong, we should still follow those outputs that are not
            //        connected to a proxy
            // FIXME: we should not need to do this anymore, as visibleDependencies should stop on proxy boundaries

            /* bool proxied = */ pmatch.visit(currentNode);
            /*
            if (proxied) {
                E_DEBUG(ENetwork, "  - Connected to a SourceProxy, not following anymore (should be handled by the composite itself, not its inner nodes)");
                continue;
            }
            */


            FNodeVector deps = currentNode->addVisibleDependencies(algoNodeMap);
            for (int i=0; i<(int)deps.size(); i++) {
              E_DEBUG(ENetwork, "  - " << deps[i]->algorithm()->name());
              toVisit.push(deps[i]);
            }
          }

          FNodeVector visibleNodes = depthFirstSearch(stepRoot);
          E_DEBUG(ENetwork, "  --2-- expand nodes");
          expandNodes(visibleNodes);
          E_DEBUG(ENetwork, "  --3-- connect expanded nodes");
          connectExpandedNodes(visibleNodes);
          E_DEBUG(ENetwork, "--------------------- CHAIN DONE ----------------------------");

        }
        else if (pstep.type() == "single") {
          E_DEBUG(ENetwork, "---------------------------------------------");
          E_DEBUG(ENetwork, "  SINGLE SHOT FOR " << pstep.algorithm()->name());
          pmatch.visit(stepRoot);

          stepRoot->expanded = expandNode(stepRoot);
          E_DEBUG(ENetwork, "------------ SINGLE SHOT DONE ---------------");
        }
        else {
          throw EssentiaException("Unknown process order step: ", pstep.type());
        }
      }

      FractalNode* expanded = stepRoot->expanded;

      //  attach nodes to deps of previous steps
      for (int i=0; i<(int)previousLeaves.size(); i++) {
        previousLeaves[i]->addChild(expanded);
      }

      previousLeaves.clear();
      FNodeVector expandedNodes = depthFirstSearch(expanded);
      for (int i=0; i<(int)expandedNodes.size(); i++) {
        if (expandedNodes[i]->children().size() == 0) previousLeaves.push_back(expandedNodes[i]);
      }
    }

    // fill inner map of the expanded algo with the proxymap values
    E_DEBUG(ENetwork, "---- EXPANDED " << calgo->name() << " LEAVES: ");
    for (int i=0; i<(int)previousLeaves.size(); i++) E_DEBUG(ENetwork, "  " << previousLeaves[i]->algorithm()->name());
    E_DEBUG(ENetwork, "---- EXPANDED " << calgo->name() << " INNER MAP ----");
    pmatch.printMatches();


    FractalNode* result = froot->children()[0];
    delete froot;

    // FIXME: not leaves, as they are not necessarily the ones that have the proxy
    result->innerMap = pmatch.proxyMap(previousLeaves);

    // clean up all the visible nodes we used before. We can only do this now because
    // we needed them before to be able to compute the proxy map correctly
    for (int i=0; i<(int)sroots.size(); i++) {
      vector<FractalNode*> stree = depthFirstSearch(sroots[i]);
      for (int j=0; j<(int)stree.size(); j++) delete stree[j];
    }

    E_DEBUG_OUTDENT;

    return result;
  }
  else {
    // standard (non-composite) algorithm: all the inner connections are on the algorithm itself
    E_DEBUG_OUTDENT;
    return expandNonCompositeNode(node);
  }
}


void Network::buildExecutionNetwork() {
  E_DEBUG(ENetwork, "building execution network");
  clearExecutionNetwork();

  // 1- First build the visible network
  E_DEBUG(ENetwork, "  1- build visible network");
  E_DEBUG_INDENT;

  FractalNode* executionNetworkRoot = visibleNetwork<FractalNode>(_generator);

  FNodeVector visibleNodes = depthFirstSearch(executionNetworkRoot);

  // 2- Expand all the nodes of this first graph
  E_DEBUG_OUTDENT;
  E_DEBUG(ENetwork, "  2- expand nodes");
  E_DEBUG_INDENT;

  expandNodes(visibleNodes);

  // 3- connect the expanded versions of the nodes together
  E_DEBUG_OUTDENT;
  E_DEBUG(ENetwork, "  3- connect expanded network");
  E_DEBUG_INDENT;
  connectExpandedNodes(visibleNodes);

  // 4- construct our "clean" execution network and clean up the FractalNodes
  E_DEBUG_OUTDENT;
  E_DEBUG(ENetwork, "  4- construct final network");
  E_DEBUG_INDENT;
  FNodeVector expandedNodes = depthFirstSearch(executionNetworkRoot->expanded);
  E_DEBUG(ENetwork, "num connected expanded nodes: " << expandedNodes.size());
  map<FractalNode*, NetworkNode*> falgoMap; // expanded → final network node

  for (int i=0; i<(int)expandedNodes.size(); i++) {
    falgoMap[expandedNodes[i]] = new NetworkNode(expandedNodes[i]->algorithm());
  }
  for (int i=0; i<(int)expandedNodes.size(); i++) {
    NetworkNode* parent = falgoMap[expandedNodes[i]];
    vector<FractalNode*> children = expandedNodes[i]->children();
    for (int j=0; j<(int)children.size(); j++) {
      E_DEBUG(ENetwork, "  -  " << parent->algorithm()->name() << " → " << falgoMap[children[j]]->algorithm()->name());
      parent->addChild(falgoMap[children[j]]);
    }
  }

  _executionNetworkRoot = falgoMap[executionNetworkRoot->expanded];

  // delete the FractalNodes which we just used temporarily for building the network
  E_DEBUG(ENetwork, "cleaning up temp visible fractal nodes");
  for (int i=0; i<(int)visibleNodes.size(); i++) delete visibleNodes[i];
  E_DEBUG(ENetwork, "cleaning up temp expanded fractal nodes");
  for (int i=0; i<(int)expandedNodes.size(); i++) delete expandedNodes[i];

  E_DEBUG_OUTDENT;
  E_DEBUG(ENetwork, "execution network ok");

}


// NB: when a network is created, it should take possession of the algorithms, and
// inhibit people changing the connections after that. This could be achieved if algorithms
// would have a pointer to a network, and if not null, then those algorithms are immutable





void Network::topologicalSortExecutionNetwork() {
  // Note: we don't need to do a full-fledged topological sort here, as we do not
  // have any DAG, we actually have a dependency tree. This way we can just do a
  // depth-first search, with ref-counting to account for diamond shapes in the tree.
  // this is similar to the wavefront design pattern used in parallelization

  // Using DFS here also has the advantage that it makes as much as possible use
  // of cache locality

  // 1- get all the nodes and count the number of refs they have
  NodeVector nodes = depthFirstSearch(_executionNetworkRoot);
  map<NetworkNode*, int> refs;

  // this initialization should be useless, but let's do it anyway for clarity
  for (int i=0; i<(int)nodes.size(); i++) refs[nodes[i]] = 0;

  // count the number of refs for each node
  for (int i=0; i<(int)nodes.size(); i++) {
    const NodeVector& children = nodes[i]->children();
    for (int j=0; j<(int)children.size(); j++) {
      refs[children[j]] += 1;
    }
  }

  // 2- do DFS again, manually this time and only visit node which have no refs anymore
  _toposortedNetwork.clear();

  NodeStack toVisit;
  toVisit.push(_executionNetworkRoot);
  refs[_executionNetworkRoot] = 1;

  while (!toVisit.empty()) {
    NetworkNode* currentNode = toVisit.top();
    toVisit.pop();

    if (--refs[currentNode] == 0) {
      _toposortedNetwork.push_back(currentNode->algorithm()); // keep this node, it is good

      const NodeVector& children = currentNode->children();
      for (int i=0; i<(int)children.size(); i++) {
        toVisit.push(children[i]);
      }
    }
  }

  E_DEBUG(ENetwork, "-------------------------------------------------------------------------------------------");
  for (int i=0; i<(int)_toposortedNetwork.size(); i++) {
    E_DEBUG_NONL(ENetwork, " → " << _toposortedNetwork[i]->name());
  }
  E_DEBUG(ENetwork, ""); // for adding a newline
  E_DEBUG(ENetwork, "-------------------------------------------------------------------------------------------");
}


void Network::checkConnections() {
  vector<Algorithm*> algos = depthFirstMap(_visibleNetworkRoot, returnAlgorithm);

  for (int i=0; i<(int)algos.size(); i++) {
    Algorithm* algo = algos[i];
    for (Algorithm::OutputMap::const_iterator output = algo->outputs().begin();
         output != algo->outputs().end();
         ++output) {

      vector<SinkBase*>& sinks = output->second->sinks();

      if (sinks.empty()) {
        ostringstream msg;
        msg << output->second->fullName() << " is not connected to any sink...";
        throw EssentiaException(msg);
      }
    }
  }
}


void Network::printBufferFillState() {
  if (!E_ACTIVE(EScheduler)) return;

  vector<Algorithm*> algos = depthFirstMap(_executionNetworkRoot, returnAlgorithm);

  for (int i=0; i<(int)algos.size(); i++) {
    Algorithm* algo = algos[i];
    E_DEBUG(EScheduler, pad(algo->name(), 25) << "(called " << algo->nProcess << " times)");
    for (Algorithm::OutputMap::const_iterator output = algo->outputs().begin();
         output != algo->outputs().end();
         ++output) {

      BufferInfo buf = output->second->bufferInfo();
      const string& name = output->first;
      int available = output->second->available();
      int used = buf.size - available;
      int percent = 100 * used / buf.size;
      E_DEBUG(EScheduler, "  - " << pad(name, 24)
              << " fill " << pad(percent, 3, ' ', true) << "%   |  "
              << pad(used, 6, ' ', true) << " / " << pad(buf.size, 6)
              << "  |  contiguous: " << pad(buf.maxContiguousElements, 6)
              << "  |  total produced: " << pad(output->second->totalProduced(), 8));
      // if we compile without debugging
      NOWARN_UNUSED(name);
      NOWARN_UNUSED(percent);
    }
    E_DEBUG(EScheduler, "");
  }
}

void printNetworkBufferFillState() {
  if (!Network::lastCreated) {
    E_WARNING("No network created, or last created network has been deleted...");
  }

  Network::lastCreated->printBufferFillState();
}

bool isExcludedFromInfo(const string& algoname) {
  // list of algorithms for which we don't want to log a resize of
  // the buffer on the info stream (still gets on the debug stream)
  static const char* excluded[2] = { "VectorInput", "Envelope" };

  for (int i=0; i<(int)ARRAY_SIZE(excluded); i++) {
    if (algoname == excluded[i]) {
      return true;
    }
  }
  return false;
}

void Network::checkBufferSizes() {
  // TODO: we should do this on the execution network, right?
  E_DEBUG(ENetwork, "checking buffer sizes");
  vector<Algorithm*> algos = depthFirstMap(_executionNetworkRoot, returnAlgorithm);

  for (int i=0; i<(int)algos.size(); i++) {
    Algorithm* algo = algos[i];

    for (Algorithm::OutputMap::const_iterator output = algo->outputs().begin();
         output != algo->outputs().end();
         ++output) {

      SourceBase* source = output->second;
      vector<SinkBase*>& sinks = source->sinks();

      BufferInfo sbuf = source->bufferInfo();
      bool noInfo = isExcludedFromInfo(source->parent()->name());

      if (sbuf.maxContiguousElements + 1 < source->acquireSize()) {
        if (noInfo) {
          E_DEBUG(EAlgorithm, "On source " << source->fullName() << ":");
          E_DEBUG(EAlgorithm, "BUFFER SIZE MISMATCH: max=" << sbuf.maxContiguousElements
                  << " - asked for write size " << source->acquireSize());
          sbuf.maxContiguousElements = (int)(source->acquireSize() * 1.1);
          sbuf.size = 8 * sbuf.maxContiguousElements;
          E_DEBUG(EAlgorithm, "resizing buffer to " << sbuf.size << "/" << sbuf.maxContiguousElements);
        }
        else {
          E_INFO("On source " << source->fullName() << ":");
          E_INFO("BUFFER SIZE MISMATCH: max=" << sbuf.maxContiguousElements
                 << " - asked for write size " << source->acquireSize());
          sbuf.maxContiguousElements = (int)(source->acquireSize() * 1.1);
          sbuf.size = 8 * sbuf.maxContiguousElements;
          E_INFO("resizing buffer to " << sbuf.size << "/" << sbuf.maxContiguousElements);
        }
      }

      for (vector<SinkBase*>::iterator it = sinks.begin(); it!=sinks.end(); ++it) {
        SinkBase* sink = *it;

        if (sbuf.maxContiguousElements + 1 < sink->acquireSize()) {
          if (noInfo) {
            E_DEBUG(EAlgorithm, "On connection " << source->fullName() << " → " << sink->fullName() << ":");
            E_DEBUG(EAlgorithm, "BUFFER SIZE MISMATCH: max=" << sbuf.maxContiguousElements
                   << " - asked for read size " << sink->acquireSize());
            sbuf.maxContiguousElements = (int)(sink->acquireSize() * 1.1);
            sbuf.size = 8 * sbuf.maxContiguousElements;
            E_DEBUG(EAlgorithm, "resizing buffer to " << sbuf.size << "/" << sbuf.maxContiguousElements);
          }
          else {
            E_INFO("On connection " << source->fullName() << " → " << sink->fullName() << ":");
            E_INFO("BUFFER SIZE MISMATCH: max=" << sbuf.maxContiguousElements
                   << " - asked for read size " << sink->acquireSize());
            sbuf.maxContiguousElements = (int)(sink->acquireSize() * 1.1);
            sbuf.size = 8 * sbuf.maxContiguousElements;
            E_INFO("resizing buffer to " << sbuf.size << "/" << sbuf.maxContiguousElements);
          }
        }
      }
      source->setBufferInfo(sbuf);
    }
  }
  E_DEBUG(ENetwork, "checking buffer sizes ok");
}


} // namespace scheduler
} // namespace essentia
