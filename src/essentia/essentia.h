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

#ifndef ESSENTIA_ESSENTIA_H
#define ESSENTIA_ESSENTIA_H

#include "config.h"
#include "types.h"

// the following are not necessary but are here for commodity purposes,
// ie: so people can just include "essentia.h" for the basic types and operations
// and then do not need stringutil.h and streamutil.h, etc... which they might
// forget otherwise or not even be aware of
#include "essentiautil.h"
#include "stringutil.h"
#include "streamutil.h"


namespace essentia {

extern const char* version;
extern const char* version_git_sha;

/**
 * This function registers the algorithms in the factory, so that they are
 * ready for use. It also builds a list of available types and their
 * "human-readable" representations. You need to call this function before
 * doing anything with essentia.
 */
void init();

bool isInitialized();

void shutdown();

namespace standard {
  /**
   * This function registers the algorithms in the factory. The waf build script 
   * dynamically generates the contents of the file essentia_algorithms_reg.cpp
   * which implements this function.
   */
  void ESSENTIA_API registerAlgorithm();
}

namespace streaming {
  /**
   * This function registers the algorithms in the factory. The waf build script 
   * dynamically generates the contents of the file essentia_algorithms_reg.cpp
   * which implements this function.
   */
  void ESSENTIA_API registerAlgorithm();
}

class TypeMap {
 public:

  static EssentiaMap<std::string, std::string>& instance() {
    if (!_typeMap) {
      throw EssentiaException("Essentia TypeMap not initialised!");
    }
    return *_typeMap;
  }

  static void init() {
    if (_typeMap) return;

    _typeMap = new EssentiaMap<std::string, std::string>();

#define registerEssentiaType(type) TypeMap::_typeMap->insert(typeid(type).name(), #type)

    registerEssentiaType(std::string);
    registerEssentiaType(Real);
    registerEssentiaType(StereoSample);
    registerEssentiaType(int);
    registerEssentiaType(uint);
    registerEssentiaType(long);
    registerEssentiaType(std::vector<std::string>);
    registerEssentiaType(std::vector<Real>);
    registerEssentiaType(std::vector<StereoSample>);

#undef registerEssentiaType
  }

  static void shutdown() {
    delete _typeMap;
    _typeMap = 0;
  }

protected:
  TypeMap() {}
  static EssentiaMap<std::string, std::string>* _typeMap;
};

} // namespace essentia

#endif // ESSENTIA_ESSENTIA_H
