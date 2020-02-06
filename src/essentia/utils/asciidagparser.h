/*
 * Copyright (C)  2006-2020  Music Technology Group - Universitat Pompeu Fabra
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
