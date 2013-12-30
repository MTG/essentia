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

#ifndef ESSENTIA_ATOMIC_H
#define ESSENTIA_ATOMIC_H


// life's easy in C++11
#if __cplusplus >= 201103L


#include <atomic>

namespace essentia {
typedef std::atomic<int> Atomic;
}


#elif defined(OS_WIN32)


#include <windows.h>

namespace essentia {

class Atomic {
 private:
  LONG volatile i_;

 public:
  inline Atomic(const int &i = 0) : i_(i) {}

  inline operator int() const { return i_; }

  inline void operator-=(const int &i) {
    InterlockedExchangeAdd(&i_, -i);
  }

  inline void operator+=(const int &i) {
    InterlockedExchangeAdd(&i_, i);
  }

  inline void operator++() {
    InterlockedIncrement(&i_);
  }

  inline void operator--() {
    InterlockedDecrement(&i_);
  }
};

} // namespace essentia


#elif defined(OS_LINUX)


#include <ext/atomicity.h>

namespace essentia {

class Atomic {
 public:
  _Atomic_word _a;

  inline Atomic(const int &i = 0) : _a(i) {}

  inline operator int () const { return _a; }

  inline void add(const int& i) {
// not sure 4.0 is the correct version, it happened somewhere between 3.3 and 4.1
#if GCC_VERSION >= 40000
    __gnu_cxx::__atomic_add(&_a,i);
#else
    __atomic_add(&_a, i);
#endif
  }

  inline void operator-=(const int &i) { add(-i); }
  inline void operator+=(const int &i) { add(i); }

  inline void operator++() { add(1); }
  inline void operator--() { add(-1); }
};

} // namespace essentia


#endif

#endif // ESSENTIA_ATOMIC_H
