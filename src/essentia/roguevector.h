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

// Windows implementation
#ifdef OS_WIN32

template <typename T>
class RogueVector : public std::vector<T> {
 protected:
  bool _ownsMemory;

 public:
  RogueVector(T* tab = 0, uint size = 0) : std::vector<T>(), _ownsMemory(false) {
    setData(tab);
    setSize(size);
  }

  RogueVector(uint size, T value) : std::vector<T>(size, value), _ownsMemory(true) {}

  ~RogueVector() {
    if (!_ownsMemory) {
      setData(0);
      setSize(0);
    }
  }

  inline void setData(T* data) { this->_Myfirst = data; }

  inline void setSize(size_t size) {
    this->_Mylast = this->_Myfirst + size;
    this->_Myend = this->_Myfirst + size;
  }

};

#endif // OS_WIN32



// Linux implementation
#if defined(OS_LINUX) || defined (OS_MAC)

// trick to detect g++ version, along with libstdc++ version
// on the cluster, __GXX_ABI_VERSION = 102 (should correspond to GCC 3.3)
#if __GXX_ABI_VERSION == 102

template <typename T>
class RogueVector : public std::vector<T> {
 protected:
  bool _ownsMemory;

 public:
  RogueVector(T* tab = 0, uint size = 0) : std::vector<T>(), _ownsMemory(false) {
    setData(tab);
    setSize(size);
  }

  RogueVector(uint size, T value) : std::vector<T>(size, value), _ownsMemory(true) {}

  ~RogueVector() {
    if (!_ownsMemory) {
      setData(0);
      setSize(0);
    }
  }

  inline void setData(T* data) { this->_M_start = data; }

  inline void setSize(size_t size) {
    this->_M_finish = this->_M_start + size;
    this->_M_end_of_storage = this->_M_start + size;
  }

};

#else // __GXX_ABI_VERSION == 102

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
    setData(v._M_impl._M_start);
    setSize(v.size());
  }

  ~RogueVector() {
    if (!_ownsMemory) {
      setData(0);
      setSize(0);
    }
  }

  inline void setData(T* data) { this->_M_impl._M_start = data; }

  inline void setSize(size_t size) {
    this->_M_impl._M_finish = this->_M_impl._M_start + size;
    this->_M_impl._M_end_of_storage = this->_M_impl._M_start + size;
  }
};

#endif // __GXX_ABI_VERSION == 102

#endif // OS_LINUX


} // namespace essentia

#endif // ESSENTIA_ROGUEVECTOR_H
