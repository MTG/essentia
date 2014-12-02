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

#ifndef ESSENTIA_UTILS_ASCIIDAG_H
#define ESSENTIA_UTILS_ASCIIDAG_H

#include <vector>
#include <string>
#include "types.h"

namespace essentia {

/**
 * Represents a position in an integer 2D plane.
 */
class Position {
 public:
  int x, y;

  Position(int x_, int y_) : x(x_), y(y_) {}

  Position operator+(const Position& other) const {
    return Position(x+other.x, y+other.y);
  }

  bool operator==(const Position& other) const {
    return x == other.x && y == other.y;
  }
};

inline std::ostream& operator<<(std::ostream& out, const Position& pos) {
  return out << '<' << pos.x << ',' << pos.y << '>';
}

/** Represents a direction when moving in an integer 2D plane */
typedef Position Direction;



/**
 * Take a string representing the network in ASCII art form and turn it into a
 * rectangle, which is a vector of string rows where each row has the same length.
 * This will allow us to navigate in it with rect[x][y], for instance.
 */
std::vector<std::string> makeRectangle(const std::string& network);

/**
 * Take an array of C strings representing the network in ASCII art form and turn it into a
 * rectangle, which is a vector of string rows where each row has the same length.
 * This will allow us to navigate in it with rect[x][y], for instance.
 */
std::vector<std::string> makeRectangle(const char* const* network, int size);


/**
 * This class represents an ASCII canvas which can contain anything. It is
 * represented as a matrix of @c chars. The only restriction (guarantee?) is
 * that every row (line) has the same number of columns (same length).
 */
class AsciiCanvas : public std::vector<std::string> {
 public:
  // NB: template is only used so that ARRAY_SIZE can work, we only want const char*[] here
  template <typename NetworkType>
  explicit AsciiCanvas(const NetworkType& network) {
    (*this) = makeRectangle(network, ARRAY_SIZE(network));
  }

  AsciiCanvas& operator=(const std::vector<std::string>& other);

  int height() const { return (int)size(); }
  int width() const { return (height() == 0) ? 0 : at(0).size(); }

  /**
   * Add an empty border en each of the 4 sides.
   * This might be useful for having to deal with less bounds-checking.
   */
  void addBorder();

  /**
   * Fill the canvas with the given char
   */
  void fill(char c);

  const std::string& at(int i) const { return std::vector<std::string>::at(i); }
        std::string& at(int i)       { return std::vector<std::string>::at(i); }

  char  at(const Position& pos) const { return (*this)[pos.y][pos.x]; }
  char& at(const Position& pos)       { return (*this)[pos.y][pos.x]; }
};

inline std::ostream& operator<<(std::ostream& out, const AsciiCanvas& canvas) {
  out << '\n';
  for (int i=0; i<canvas.height(); i++) out << canvas.at(i) << '\n';
  return out;
}


/**
 * This class represents a box in an ASCII canvas.
 * At the moment, the only chars allowed for drawing boxes are:
 *  - '+' for corners
 *  - '-' for horizontal lines
 *  - '|' for vertical lines
 */
class AsciiBox {
 public:
  int posX, posY;    // position of the top-left corner
  int width, height; // size of the box
  std::string title; // title of the box (main name contained in it)

  /**
   * This constructs a box that is supposedly anchored at pos (x, y) in the given
   * network. If that is not the case, the result is undefined.
   */
  AsciiBox(const std::vector<std::string>& network, int x, int y);

  /**
   * Return whether the given position is located on the border (frame) of this box.
   */
  bool borderContains(int x, int y) const;

  /**
   * Return whether there is a box which top-left corner is located at (x, y).
   */
  static bool isBox(const std::vector<std::string>& network, int x, int y);

  /**
   * Find and return all the boxes in the given ascii network representation.
   * This function does not do any checking of any sort, so an ill-formed box will
   * simply be ignored and no exception will be thrown
   */
  static std::vector<AsciiBox> findBoxes(const std::vector<std::string>& network);

};


} // namespace essentia

#endif // ESSENTIA_UTILS_ASCIIDAG_H
