/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_ROGUEVECTOR_H
#define ESSENTIA_ROGUEVECTOR_H

#include <vector>
#include "types.h"

namespace essentia {


template <typename T>
class RogueVector : public std::vector<T> {
 protected:
  bool _ownsMemory;

 public:
  RogueVector(T* tab = 0, size_t size = 0) : std::vector<T>(), _ownsMemory(false) {
    setData(tab);
    setSize(size);
  }

  RogueVector(uint size, T value) : std::vector<T>(size, value), _ownsMemory(true) {}

  RogueVector(const RogueVector<T>& v) : std::vector<T>(), _ownsMemory(false) {
    setData(const_cast<T*>(v.data()));
    setSize(v.size());
  }

  ~RogueVector() {
    if (!_ownsMemory) {
      setData(0);
      setSize(0);
    }
  }

  // Those need to be implementation specific
  void setData(T* data);
  void setSize(size_t size);
};

// Clang/LLVM implementation
#if defined(__clang__) || defined(__EMSCRIPTEN__)

// TODO: this is a big hack that relies on clang/libcpp not changing the memory
//       layout of the std::vector (very dangerous, but works for now...)

template <typename T>
void RogueVector<T>::setData(T* data) { *reinterpret_cast<T**>(this) = data; }

template <typename T>
void RogueVector<T>::setSize(size_t size) {
    T** start = reinterpret_cast<T**>(this);
    *(start+1) = *start + size;
    *(start+2) = *start + size;
}

// Linux implementation
#elif defined(OS_LINUX) || defined(OS_MINGW)

template <typename T>
void RogueVector<T>::setData(T* data) { this->_M_impl._M_start = data; }

template <typename T>
void RogueVector<T>::setSize(size_t size) {
  this->_M_impl._M_finish = this->_M_impl._M_start + size;
  this->_M_impl._M_end_of_storage = this->_M_impl._M_start + size;
}

// Windows / MSVC implementation, VS2017 and earlier.
// _Myfirst()/_Mylast()/_Myend() are accessible protected member functions
// on std::vector's internal base classes on these toolsets.
#elif defined(OS_WIN32) && defined(_MSC_VER) && _MSC_VER <= 1916

template <typename T>
void RogueVector<T>::setData(T* data) {
  this->_Myfirst() = data;
}

template <typename T>
void RogueVector<T>::setSize(size_t size) {
  this->_Mylast() = this->_Myfirst() + size;
  this->_Myend() = this->_Myfirst() + size;
}

// Windows / MSVC implementation, VS2019 and later.
// VS 2019+ restructured std::vector internals so _Myfirst()/_Mylast()/_Myend()
// are no longer accessible from derived classes.  Fall back to the same
// raw-pointer approach as the Clang branch above: MSVC's std::vector stores
// three consecutive T* (_Myfirst, _Mylast, _Myend), with EBO collapsing the
// empty std::allocator<T>.
//
// When iterator debugging is enabled (_ITERATOR_DEBUG_LEVEL != 0, the
// default for Debug configurations), MSVC's STL additionally prepends a
// hidden _Container_proxy* as the very first member of the vector object
// (used to track debug iterators), which shifts the three data pointers one
// slot to the right. Skipping that offset corrupts the vector object -- the
// write lands on the proxy pointer instead of _Myfirst -- and the vector
// triggers a "vector subscript out of range" assertion the moment it is
// next used. See https://github.com/MTG/essentia/pull/1514.
#elif defined(OS_WIN32) && defined(_MSC_VER)

template <typename T>
void RogueVector<T>::setData(T* data) {
#if _ITERATOR_DEBUG_LEVEL != 0
  *(reinterpret_cast<T**>(this) + 1) = data;
#else
  *reinterpret_cast<T**>(this) = data;
#endif
}

template <typename T>
void RogueVector<T>::setSize(size_t size) {
#if _ITERATOR_DEBUG_LEVEL != 0
  T** start = reinterpret_cast<T**>(this) + 1;
#else
  T** start = reinterpret_cast<T**>(this);
#endif
  *(start+1) = *start + size;
  *(start+2) = *start + size;
}

#endif

} // namespace essentia

#endif // ESSENTIA_ROGUEVECTOR_H
