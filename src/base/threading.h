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

#ifndef ESSENTIA_THREADING_H
#define ESSENTIA_THREADING_H

namespace essentia {

// if we were to use TBB for the scheduler, we would have:
/*
typedef tbb::spin_mutex Mutex;
typedef tbb::spin_mutex::scoped_lock MutexLocker;
typedef tbb::spin_mutex ForcedMutex;
typedef tbb::spin_mutex::scoped_lock ForcedMutexLocker;
*/

// The mutex in essentia only needs to be a real mutex when it is possible
// to call the algorithms in a multithreaded way.
// If not, it can be replaced with a no-op mutex for performance reasons.

class Mutex {
 public:
  void lock() {}
  void unlock() {}
};

class MutexLocker {
 public:
  MutexLocker(Mutex& mutex) {}
  void release() {}
  void acquire(Mutex&) {}
};


// the ForcedMutex is a real Mutex, that should always lock properly
// (ex: in FFTW, the plan creation/destruction needs to be protected no matter what)

#  ifdef OS_WIN32

// windows CriticalSection implementation
#include <windows.h>

class ForcedMutex {
 protected:
  CRITICAL_SECTION criticalSection;
 public:
  ForcedMutex()  { InitializeCriticalSection(&criticalSection); }
  ~ForcedMutex() { DeleteCriticalSection(&criticalSection); }
  void lock()    { EnterCriticalSection(&criticalSection); }
  void unlock()  { LeaveCriticalSection(&criticalSection); }
};

#  else // OS_WIN32

// posix implementation for linux and osx
#include <pthread.h>

class ForcedMutex {
 protected:
  pthread_mutex_t pthreadMutex;
 public:
  ForcedMutex() {
    if (pthread_mutex_init(&pthreadMutex,0) != 0)
      throw EssentiaException("can't create mutex type");
  }
  ~ForcedMutex() { pthread_mutex_destroy(&pthreadMutex); }
  void lock()    { pthread_mutex_lock(&pthreadMutex); }
  void unlock()  { pthread_mutex_unlock(&pthreadMutex); }
};

#  endif // OS_WIN32

class ForcedMutexLocker {
 protected:
  ForcedMutex& _mutex;
 public:
  ForcedMutexLocker(ForcedMutex& mutex) : _mutex(mutex) { _mutex.lock(); }
  ~ForcedMutexLocker() { _mutex.unlock(); }
};


} // namespace essentia

#endif // ESSENTIA_THREADING_H
