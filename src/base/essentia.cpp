#include "essentia.h"
#include "algorithmfactory.h"
#include <fftw3.h>

#ifndef OS_WIN32
#include <cxxabi.h> // for __cxa_demangle
#endif

using namespace std;

namespace essentia {

const char* version = ESSENTIA_VERSION;


bool _initialized;

void init() {
  standard::AlgorithmFactory::init();
  standard::registerAlgorithm();
  streaming::AlgorithmFactory::init();
  streaming::registerAlgorithm();
  TypeMap::init();

  _initialized = true;
}

void shutdown() {
  fftwf_cleanup();
  standard::AlgorithmFactory::shutdown();
  streaming::AlgorithmFactory::shutdown();
  TypeMap::shutdown();

  _initialized = false;
}

bool isInitialized() {
  return _initialized;
}

template<> standard::AlgorithmFactory* standard::AlgorithmFactory::_instance = 0;
template<> streaming::AlgorithmFactory* streaming::AlgorithmFactory::_instance = 0;

EssentiaMap<string,string> * TypeMap::_typeMap = 0;


#ifndef OS_WIN32

string demangle(const char* name) {
  char buf[1024];
  size_t size = 1024;
  int status;
  char* res = abi::__cxa_demangle(name,
                                  buf,
                                  &size,
                                  &status);
  return res;
}

#else // OS_WIN32

string demangle(const char* name) {
  return name;
}

#endif // OS_WIN32


// declared in src/base/types.h
string nameOfType(const std::type_info& type) {
  try {
    return TypeMap::instance()[type.name()];
  }
  catch (EssentiaException&) {
    return demangle(type.name());
  }
}

} // namespace essentia
