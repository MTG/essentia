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

/**
 * Initialize Essentia and fill the AlgorithmFactories with the Algorithms.
 */
void init() {
  setDebugLevel(EUser1 | EUser2);

  E_DEBUG(EFactory, "essentia::init()");
  standard::AlgorithmFactory::init();
  standard::registerAlgorithm();
  streaming::AlgorithmFactory::init();
  streaming::registerAlgorithm();
  TypeMap::init();

  _initialized = true;
  E_DEBUG(EFactory, "essentia::init() ok!");
}

/**
 * Cleanup all resources allocated by Essentia.
 */
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
