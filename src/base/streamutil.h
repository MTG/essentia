/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
