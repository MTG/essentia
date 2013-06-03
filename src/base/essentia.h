/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
  void ESSENTIA_API registerAlgorithm();
}

namespace streaming {
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
