/*
 * Copyright (C) 2006-2010 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_UTILS_ASCIIDAGPARSER_H
#define ESSENTIA_UTILS_ASCIIDAGPARSER_H

#include "asciidag.h"

namespace essentia {


class AsciiDAGParser {
 public:
  // NB: template is only used so that ARRAY_SIZE can work, we only want const char*[] here
  template <typename NetworkType>
  AsciiDAGParser(const NetworkType& network) : _network(network) {
    _network.addBorder(); // spares us lots of checks afterwards
    parseGraph();
  }

  /**
   * Does the actual parsing of the network, get:
   *  - node names
   *  - parameters, if any
   *  - connections, possibly (un)named
   */
  void parseGraph();

  /**
   * Return the node names. They are sorted lexicographically.
   */
  const std::vector<std::string>& nodes() const { return _nodes; }


  /**
   * Return the edges. They are sorted lexicographically.
   */
  const std::vector<std::pair<std::string, std::string> >& namedEdges() const { return _namedEdges; }
  /**
   * Return the edges using node indices, to avoid ambiguities if 2 or more nodes have the same name.
   * They are sorted in the same order as the named edges.
   */
  const std::vector<std::pair<int, int> >& edges() const { return _edges; }


 protected:
  AsciiCanvas _network;
  std::vector<std::string> _nodes;
  std::vector<std::pair<int, int> > _edges; // node id -> node id
  std::vector<std::pair<std::string, std::string> > _namedEdges; // node name -> node name

  void parseEdges(const std::vector<AsciiBox>& boxes);
};

} // namespace essentia

#endif // ESSENTIA_UTILS_ASCIIDAGPARSER_H
