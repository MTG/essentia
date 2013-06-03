#ifndef ESSENTIA_ATOMIC_H
#define ESSENTIA_ATOMIC_H



#ifdef OS_WIN32

#include <windows.h>

//
// Atomic for OS_WIN32
//

namespace essentia {

class Atomic
{
private:
	LONG volatile i_;
public:
	inline Atomic(const int &i = 0) : i_(i)
	{
	}

	inline operator int () const
	{
		return i_;
	}

	inline void operator -=(const int &i)
	{
		InterlockedExchangeAdd(&i_, -i);
	}

	inline void operator +=(const int &i)
	{
		InterlockedExchangeAdd(&i_, i);
	}

	inline void operator ++()
	{
		InterlockedIncrement(&i_);
	}

	inline void operator --()
	{
		InterlockedDecrement(&i_);
	}
};

} // namespace essentia

#else // OS_WIN32

//
// Atomic for OS_MAC and OS_LINUX:
//

#if GCC_VERSION >= 40200
   // atomicity.h has moved to ext/ since g++ 4.2
  #if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ <=1050) // not on osx 10.5
  #  include <bits/atomicity.h>
  #else
  #  include <ext/atomicity.h>
  #endif
#else
#  include <bits/atomicity.h>
#endif

namespace essentia {

class Atomic
{
public:
	_Atomic_word _a;

	inline Atomic(const int &i = 0)
	:_a(i)
	{
	}

	inline operator int () const
	{
		return _a;
	}

	inline void add(const int& i)
	{
// not sure 4.0 is the correct version, it happened somewhere between 3.3 and 4.1
#if GCC_VERSION >= 40000
		__gnu_cxx::__atomic_add(&_a,i);
#else
                __atomic_add(&_a, i);
#endif
	}

	inline void operator -=(const int &i)
	{
		add(-i);
	}

	inline void operator +=(const int &i)
	{
		add(i);
	}

	inline void operator ++()
	{
		add(1);
	}

	inline void operator --()
	{
		add(-1);
	}
};

} // namespace essentia

#endif // OS_WIN32

#endif // ESSENTIA_ATOMIC_H
