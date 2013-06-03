/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_STRONGPEAK_H
#define ESSENTIA_STRONGPEAK_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class StrongPeak : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _strongPeak;

 public:
  StrongPeak() {
    declareInput(_spectrum, "spectrum", "the input spectrum (must be greater than one element and cannot contain negative values)");
    declareOutput(_strongPeak, "strongPeak", "the Strong Peak ratio");
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

class StrongPeak : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _strongPeak;

 public:
  StrongPeak() {
    declareAlgorithm("StrongPeak");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_strongPeak, TOKEN, "strongPeak");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_STRONGPEAK_H
