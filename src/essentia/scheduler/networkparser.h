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

#ifndef ESSENTIA_SCHEDULER_NETWORKPARSER_H
#define ESSENTIA_SCHEDULER_NETWORKPARSER_H

#include "network.h"
#include "../utils/asciidagparser.h"

namespace essentia {
namespace scheduler {

class NetworkParser {
 public:
  // NB: template is only used so that ARRAY_SIZE can work, we only want const char*[] here
  // if createConnections is false, only the NetworkNodes will be connected according to
  // the topology, not the underlying algorithms. This is useful for unittesting when we have
  // diamond shapes, which can appear in the execution network but not in the visible network
  template <typename NetworkType>
  NetworkParser(const NetworkType& network, bool createConnections = true) : _graph(network) {
    createNetwork(createConnections);
  }

  ~NetworkParser() {
    // no need to delete algorithms that have been instantiated as those are being
    // taken care of by the Network, which was created with takeOwnership = true.
    delete _network;
  }

  const std::vector<std::string>& algorithms() const { return _graph.nodes(); }
  const std::vector<std::pair<int, int> >& connections() const { return _graph.edges(); }
  const std::vector<std::pair<std::string, std::string> >& namedConnections() const { return _graph.namedEdges(); }

  Network* network() { return _network; }

 protected:
  AsciiDAGParser _graph;
  Network* _network;

  // from the parsed DAG, create the corresponding algorithms and connect them.
  void createNetwork(bool createConnections = true);
  void createConnections();

  // algos that have been instantiated
  std::vector<streaming::Algorithm*> _algos;

};

} // namespace scheduler
} // namespace essentia

#endif // ESSENTIA_SCHEDULER_NETWORKPARSER_H
