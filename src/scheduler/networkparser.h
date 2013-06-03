/*
 * Copyright (C) 2006-2010 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_SCHEDULER_NETWORKPARSER_H
#define ESSENTIA_SCHEDULER_NETWORKPARSER_H

#include "network.h"
#include "asciidagparser.h"

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
