/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_FLATNESSDB_H
#define ESSENTIA_FLATNESSDB_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class FlatnessDB : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<Real> _flatnessDB;

  Algorithm* _flatness;

 public:
  FlatnessDB() {
    declareInput(_array, "array", "the input array");
    declareOutput(_flatnessDB, "flatnessDB", "the flatness dB");

    _flatness = AlgorithmFactory::create("Flatness");
  }

  ~FlatnessDB() {
    if (_flatness) delete _flatness;
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

class FlatnessDB : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _flatnessDB;

 public:
  FlatnessDB() {
    declareAlgorithm("FlatnessDB");
    declareInput(_array, TOKEN, "array");
    declareOutput(_flatnessDB, TOKEN, "flatnessDB");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_FLATNESSDB_H
