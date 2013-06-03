/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_CREST_H
#define ESSENTIA_CREST_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Crest : public Algorithm {

 protected:
  Input<std::vector<Real> > _array;
  Output<Real> _crest;

 public:
  Crest() {
    declareInput(_array, "array", "the input array (cannot contain negative values, and must be non-empty)");
    declareOutput(_crest, "crest", "the crest of the input array");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Crest : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _crest;

 public:
  Crest() {
    declareAlgorithm("Crest");
    declareInput(_array, TOKEN, "array");
    declareOutput(_crest, TOKEN, "crest");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_CREST_H
