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

#include "networkparser.h"
#include <algorithm>
#include <stack>
#include <iostream>
#include "stringutil.h"
#include "algorithmfactory.h"
using namespace std;

namespace essentia {
namespace scheduler {

void NetworkParser::createNetwork(bool createConnections) {
  const vector<string>& nodes = _graph.nodes();
  const vector<pair<int, int> >& edges = _graph.edges();

  // create all algorithms corresponding to the nodes
  for (int i=0; i<(int)nodes.size(); i++) {
    const string& name = nodes[i];
    _algos.push_back(streaming::AlgorithmFactory::create(name));
  }

  // find root node. We will assume here it is the only one that is not connected
  // through its inputs
  int rootNodeIdx = -1;
  for (int i=0; i<(int)nodes.size(); i++) {
    // if this node never appears as second element of any edge, we found it!
    bool isRoot = true;
    for (int j=0; j<(int)edges.size(); j++) {
      if (edges[j].second == i) {
        isRoot = false;
        break;
      }
    }
    if (isRoot) rootNodeIdx = i;
  }

  assert(rootNodeIdx >= 0);

  // create algorithms connections if asked, otherwise connect directly the
  // corresponding NetworkNodes
  if (createConnections) {
    this->createConnections();
    _network = new Network(_algos[rootNodeIdx]);
  }
  else {
    // we need to instantiate and connect the network nodes explicitly
    map<streaming::Algorithm*, NetworkNode*> algoNodeMap;
    _network = new Network(_algos[rootNodeIdx]);
    algoNodeMap.insert(make_pair(_algos[rootNodeIdx], _network->visibleNetworkRoot()));

    for (int i=0; i<(int)_algos.size(); i++) {
      if (i == rootNodeIdx) continue;
      algoNodeMap.insert(make_pair(_algos[i], new NetworkNode(_algos[i])));
    }

    for (int i=0; i<(int)edges.size(); i++) {
      algoNodeMap[_algos[edges[i].first]]->addChild(algoNodeMap[_algos[edges[i].second]]);
    }
  }
}

void NetworkParser::createConnections() {
  // create all connections
  for (int i=0; i<(int)_graph.edges().size(); i++) {
    const pair<int, int>& edge = _graph.edges()[i];
    streaming::Algorithm* src = _algos[edge.first];
    streaming::Algorithm* dst = _algos[edge.second];

    // make sure both algorithms have the same number of inputs/outputs
    if (src->outputs().size() != dst->inputs().size()) {
      ostringstream msg;
      msg << "Cannot connect " << src->name() << " to " << dst->name()
          << " because they don't have the same number of inputs/outputs. ("
          << src->name() << ": " << src->outputs().size() << " outputs - "
          << dst->name() << ": " << dst->inputs().size() << " inputs)";
      throw EssentiaException(msg);
    }

    // connect the inputs/outputs in the order they have been defined
    for (int pidx=0; pidx<(int)src->outputs().size(); pidx++) {
      connect(src->output(pidx), dst->input(pidx));
    }
  }
}

} // namespace scheduler
} // namespace essentia
