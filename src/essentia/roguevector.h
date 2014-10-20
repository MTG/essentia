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

#if defined(__GNUC__) && GCC_VERSION < 40200
  T* data();
  const T* data() const;
#endif

};




// MSVC implementation
#if defined(_MSV_VER)


template <typename T>
void RogueVector<T>::setData(T* data) { this->_Myfirst = data; }

template <typename T>
void RogueVector<T>::setSize(size_t size) {
  this->_Mylast = this->_Myfirst + size;
  this->_Myend = this->_Myfirst + size;
}


// GCC implementation
#elif defined(__GNUC__)


template <typename T>
void RogueVector<T>::setData(T* data) { this->_M_impl._M_start = data; }

template <typename T>
void RogueVector<T>::setSize(size_t size) {
  this->_M_impl._M_finish = this->_M_impl._M_start + size;
  this->_M_impl._M_end_of_storage = this->_M_impl._M_start + size;
}

#if GCC_VERSION < 40200

template <typename T>
T* RogueVector<T>::data() {
  return this->_M_impl._M_start;
}

template <typename T>
const T* RogueVector<T>::data() const {
  return this->_M_impl._M_start;
}

#endif

// clang/libcpp implementation
#elif defined (_LIBCPP_VERSION)


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


#else

#error "unsupported STL vendor"

#endif

} // namespace essentia

#endif // ESSENTIA_ROGUEVECTOR_H
