/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
// If not, it can be replaced with a no-op mutex.

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
