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

#include "asciidagparser.h"
#include "stringutil.h"
#include <algorithm>
#include <stack>
using namespace std;

namespace essentia {


Direction up(0, -1);
Direction down(0, 1);
Direction left(-1, 0);
Direction right(1, 0);

Direction compass[] = { up, right, down, left };


/**
 * Represents a path being currently followed. It consists of the current
 * position and the direction we're moving in.
 */
class Path {
 public:
  Position pos;
  Direction dir;
  AsciiCanvas visited;

  Path(const Position& p, const Direction& d, const AsciiCanvas& canvas) : pos(p), dir(d), visited(canvas) {
    visited.fill('0');
    visited.at(pos) = '1';
  }

  Position next() const { return pos+dir; }
  void advance() {
    pos = pos + dir;
    visited.at(pos) = '1';
  }

  bool alreadyVisited(const Position& p) const {
    return visited.at(p) == '1';
  }
};


bool cmpBoxes(const AsciiBox& b1, const AsciiBox& b2) {
  return b1.title < b2.title;
}

void AsciiDAGParser::parseGraph() {
  _nodes.clear();

  vector<AsciiBox> boxes = AsciiBox::findBoxes(_network);
  sort(boxes.begin(), boxes.end(), cmpBoxes);

  for (int i=0; i<(int)boxes.size(); i++) _nodes.push_back(boxes[i].title);

  parseEdges(boxes);
  sort(_edges.begin(), _edges.end());

  // named edges will also be lexicographically sorted due to the fact that the
  // nodes were already sorted that way, and so are the edges with node indices
  _namedEdges.resize(_edges.size());
  for (int i=0; i<(int)_edges.size(); i++) {
    _namedEdges[i] = make_pair(_nodes[_edges[i].first], _nodes[_edges[i].second]);
  }
}


void AsciiDAGParser::parseEdges(const vector<AsciiBox>& boxes) {
  // for each of the algorithms, look whether there is an outgoing path, and if
  // that is the case, follow it until we reach another algorithm, which we then identify
  for (int i=0; i<(int)boxes.size(); i++) {
    for (int h=0; h<boxes[i].height; h++) {
      // is there a path going out of this algo?
      int xout = boxes[i].posX+boxes[i].width+2;
      int yout = boxes[i].posY+h+1;
      if (_network[yout][xout] == '-') {
        // stack of currently followed paths
        stack<Path> paths;
        paths.push(Path(Position(xout, yout), right, _network));

        // follow all paths until the end
        while (!paths.empty()) {
          Path& p = paths.top();
          Position pnext = p.next();

          // if we loop on ourselves, we're clearly lost... stop here then
          if (p.alreadyVisited(pnext)) {
            paths.pop();
            continue;
          }

          // we follow an horizontal line
          if ((p.dir == left || p.dir == right) &&
              (_network.at(pnext) == '-')) {
            p.advance();
            continue;
          }

          // we follow a vertical line
          if ((p.dir == up || p.dir == down) &&
              (_network.at(pnext) == '|')) {
            p.advance();
            continue;
          }

          // we arrive at a crossing ('+')
          if (_network.at(pnext) == '+') {
            Path incoming = p;
            paths.pop(); // NB: p is not valid anymore starting from here

            for (int d=0; d<4; d++) {
              if (incoming.dir == compass[(d+2)%4]) continue; // this is where we came from

              if ((_network.at(pnext + compass[d]) == '+')
                  ||
                  ((_network.at(pnext + compass[d]) == '-') && (d%2 == 1)) ||
                  ((_network.at(pnext + compass[d]) == '|') && (d%2 == 0))) {

                Path newPath(incoming);
                newPath.advance();
                newPath.dir = compass[d];
                paths.push(newPath);
              }
            }

            continue;
          }

          // we arrive at an algorithm
          if ((p.dir == right) && (_network.at(pnext) == '|')) {
            // find which algorithm this belongs to
            for (int j=0; j<(int)boxes.size(); j++) {
              if (boxes[j].borderContains(pnext.x, pnext.y)) {
                _edges.push_back(make_pair(i, j));
                break;
              }
            }

            paths.pop();
            continue;
          }

          // the path is broken, just remove it
          paths.pop();

        }
      }
    }
  }
}

} // namespace essentia
