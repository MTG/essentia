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

#ifndef ESSENTIA_UTILS_H
#define ESSENTIA_UTILS_H

#include <vector>
#include <cctype>
#include <string>
#include <cmath> // isinf, isnan
#include "utils/tnt/tnt.h"
#include "types.h"


namespace essentia {


// ARRAY_SIZE returns the size of a constant-sized C-style array
#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr)/sizeof(*(arr)))
#endif


// macro used to silence compiler warnings about otherwise valid uses of
// "unused" variables (eg: RAII, see MutexLocker for instance)
#define NOWARN_UNUSED(expr) do { (void)(expr); } while (0)


/**
 * Utility function that converts a C-style array into a C++ std::vector.
 */
template <typename T, typename Array>
std::vector<T> arrayToVector(const Array& array) {
  int size = ARRAY_SIZE(array);
  std::vector<T> result(size);
  for (int i=0; i<size; i++) {
    result[i] = array[i];
  }
  return result;
}



/**
 * Return the index of the given element inside the vector. If @c elem was not
 * found in @c v, return -1.
 */
template <typename T>
int indexOf(const std::vector<T>& v, const T& elem) {
  const int size = (int)v.size();
  for (int i=0; i<size; i++) {
    if (v[i] == elem) return i;
  }

  return -1;
}


/**
 * Returns whether a vector of a certain type contains an element or not.
 * Comparison is done using the '==' operator, so you can even use it for your
 * own types if you overload this operator.
 */
template <typename T>
bool contains(const std::vector<T>& v, const T& elem) {
  return (indexOf(v, elem) != -1);
}

inline bool contains(const std::vector<std::string>& v, const char* str) {
  return contains(v, std::string(str));
}

/**
 * Utility function to test whether a key is in a map.
 */
template <typename T, typename U>
bool contains(const std::map<T, U>& m, const T& key) {
  return m.find(key) != m.end();
}

template <typename T>
bool contains(const OrderedMap<T>& m, const std::string& key) {
  const int size = (int)m.size();
  for (int i=0; i<size; i++) {
    if (m[i].first == key) return true;
  }
  return false;
}

template <typename T>
bool contains(const OrderedMap<T>& m, const char* key) {
  return contains(m, std::string(key));
}



/**
 * Return whether a value is valid or not, ie: is not Inf nor NaN.
 */
template <typename T>
inline bool isValid(const T& value) {
  if (std::isinf(value) || std::isnan(value)) return false;
  return true;
}

template <typename T>
inline bool isValid(const Tuple2<T>& value) {
  if (!isValid(value.left()) ||
      !isValid(value.right())) {
    return false;
  }
  return true;
}

inline bool isValid(const std::string& s) {
  // at the moment all strings are valid
  return true;
}

template <typename T>
inline bool isValid(const std::vector<T>& v) {
  typename std::vector<T>::const_iterator it = v.begin();
  while (it != v.end() && isValid(*it)) ++it;
  return it == v.end();
}

template <typename T>
inline bool isValid(const std::vector<std::vector<T> >& mat) {
  typename std::vector<std::vector<T> >::const_iterator it = mat.begin();
  while (it != mat.end() && isValid(*it)) ++it;
  return it == mat.end();
}

template <typename T>
inline bool isValid(const TNT::Array2D<T> & mat) {
  for (int row=0; row<mat.dim1(); ++row) {
    for (int col=0; col<mat.dim2(); ++col) {
      if (!isValid(mat[row][col])) return false;
    }
  }
  return true;
}

#ifdef OS_WIN32
int mkstemp(char *tmpl);
#endif // OS_WIN32




/**
 * This function, which has the same interface as memcpy, performs either memcpy (fast)
 * for basic types, or does a simple copy element by element (for strings, for instance).
 */
template <typename T>
inline void fastcopy(T* dest, const T* src, int n) {
  for (int i=0; i<n; i++) {
    *dest++ = *src++;
  }
}

template <>
inline void fastcopy<Real>(Real* dest, const Real* src, int n) {
  memcpy(dest, src, n*sizeof(Real));
}

// overload for iterators, which allow us to fastcopy(dest.begin(), src.begin(), 0) and not crash
inline void fastcopy(std::vector<Real>::iterator dest, std::vector<Real>::const_iterator src, int n) {
  // need to test this because otherwise it is not legal to dereference the iterator
  if (n > 0) {
    fastcopy(&*dest, &*src, n);
  }
}

template <>
inline void fastcopy<StereoSample>(StereoSample* dest, const StereoSample* src, int n) {
  memcpy(dest, src, n*sizeof(StereoSample));
}

template <>
inline void fastcopy<int>(int* dest, const int* src, int n) {
  memcpy(dest, src, n*sizeof(int));
}




} // namespace essentia

#endif // ESSENTIA_UTILS_H
