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

#include "asciidag.h"
#include "stringutil.h"
using namespace std;

namespace essentia {

AsciiCanvas& AsciiCanvas::operator=(const std::vector<std::string>& other) {
  vector<string>::operator=(other);
  return *this;
}

void AsciiCanvas::addBorder() {
  for (int i=0; i<(int)size(); i++) {
    at(i) = ' ' + at(i) + ' ';
  }
  insert(begin(), string(at(0).size(), ' '));
  push_back(string(at(0).size(), ' '));
}

void AsciiCanvas::fill(char c) {
  for (int i=0; i<height(); i++) {
    for (int j=0; j<width(); j++) {
      at(i)[j] = c;
    }
  }
}


vector<string> makeRectangle(const char* const* network, int size) {
  vector<string> result(size);
  if (size == 0) return result;

  int maxl = -1;

  for (int i=0; i<size; i++) {
    result[i] = network[i];
    maxl = max(maxl, (int)result[i].size());
  }

  for (int i=0; i<size; i++) result[i].resize(maxl, ' ');

  return result;
}

vector<string> makeRectangle(const string& network) {
  vector<string> result = tokenize(network, "\n", true);

  if (result.empty()) return result;

  int nrows = result.size();
  int maxl = result[0].size();
  for (int i=1; i<nrows; i++) maxl = max(maxl, (int)result[i].size());

  for (int i=0; i<nrows; i++) result[i].resize(maxl, ' ');

  return result;
}

inline bool isIn(int x, int a, int b) {
  return (x >= a) && (x < b);
}


AsciiBox::AsciiBox(const vector<string>& network, int x, int y)
  : posX(x), posY(y), width(0), height(0)
{
  int nrows = network.size();
  int ncols = network[0].size();

  // get box geometry
  while ((x+width+1 < ncols) && (network[y][x+width+1] == '-')) width++;
  while ((y+height+1 < nrows) && (network[y+height+1][x] == '|')) height++;

  // get box title
  title = strip(network[y+1].substr(x+1, width));
}

bool AsciiBox::borderContains(int x, int y) const {
  return (((isIn(y, posY, posY+height+2)) && ((x == posX) ||
                                              (x == posX+width+1)))
          ||
          ((isIn(x, posX, posX+width+2)) && ((y == posY) ||
                                             (y == posY+height+1))));
}


bool AsciiBox::isBox(const vector<string>& network, int x, int y) {
  int nrows = network.size();
  int ncols = network[0].size();

  // preliminary bounds-checking
  if (!isIn(x, 0, ncols) || !(isIn(y, 0, nrows))) return false;

  // we need to start with a corner for it to be a box
  if (network[y][x] != '+') return false;

  // find possible width and height
  int width = 0, height = 0;

  while ((x+width+1 < ncols) && (network[y][x+width+1] == '-')) width++;
  if ((x+width+1 == ncols) || (network[y][x+width+1] != '+')) return false;

  while ((y+height+1 < nrows) && (network[y+height+1][x] == '|')) height++;
  if ((y+height+1 == nrows) || (network[y+height+1][x] != '+')) return false;

  // now that we have a probable width and height, make sure the other half of
  // the rect is also correct
  for (int i=0; i<width; i++) if (network[y+height+1][x+i+1] != '-') return false;
  for (int i=0; i<height; i++) if (network[y+i+1][x+width+1] != '|') return false;
  if (network[y+height+1][x+width+1] != '+') return false;

  return true;
}

vector<AsciiBox> AsciiBox::findBoxes(const vector<string>& network) {
  int nrows = network.size();
  int ncols = network[0].size();
  vector<AsciiBox> result;

  for (int y=0; y<nrows; y++) {
    for (int x=0; x<ncols; x++) {
      if (isBox(network, x, y)) {
        result.push_back(AsciiBox(network, x, y));
      }
    }
  }

  return result;
}


} // namespace essentia
