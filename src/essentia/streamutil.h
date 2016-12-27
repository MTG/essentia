/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_STREAMUTIL_H
#define ESSENTIA_STREAMUTIL_H

#include <vector>
#include <set>
#include <complex>

//
// This file contains helper functions and macros to make debugging and working
// with streams easier.
//

/**
 * Prints a variable name followed by its value on a line.
 */
#define PR(x) std::cout << #x << ": " << x << std::endl


namespace essentia {


/**
 * Output a std::complex into an output stream.
 */
/*
std::ostream& operator<<(std::ostream& out, const std::complex<float>& c) {
  return out << c.real() << '+' << c.imag() << 'i';
}
*/
/**
 * Output a std::pair into an output stream.
 */
template <typename T, typename U>
std::ostream& operator<<(std::ostream& out, const std::pair<T, U>& p) {
  return out << '<' << p.first << ',' << p.second << '>';
}

/**
 * Output a std::vector into an output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << '['; if (!v.empty()) {
    out << *v.begin(); typename std::vector<T>::const_iterator it = v.begin();
    for (++it; it != v.end(); ++it) out << ", " << *it;
  }
  return out << ']';
}

/**
 * Output a std::set into an output stream.
 */
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::set<T>& v) {
  out << '{'; if (!v.empty()) {
    out << *v.begin(); typename std::set<T>::const_iterator it = v.begin();
    for (++it; it != v.end(); ++it) out << ", " << *it;
  }
  return out << '}';
}


} // namespace essentia

#endif // ESSENTIA_STREAMUTIL_H
