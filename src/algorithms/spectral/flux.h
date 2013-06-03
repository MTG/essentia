/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_FLUX_H
#define ESSENTIA_FLUX_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Flux : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _flux;

  std::vector<Real> _spectrumMemory;
  std::string _norm;
  bool _halfRectify;

 public:
  Flux() {
    declareInput(_spectrum, "spectrum", "the input spectrum");
    declareOutput(_flux, "flux", "the spectral flux of the input spectrum");
  }

  void declareParameters() {
    declareParameter("norm", "the norm to use for difference computation", "{L1,L2}", "L2");
    declareParameter("halfRectify", "half-rectify the differences in each spectrum bin", "{true,false}", false);
  }

  void configure();
  void compute();

  void reset() {
    _spectrumMemory.clear();
  }

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Flux : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _flux;

 public:
  Flux() {
    declareAlgorithm("Flux");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_flux, TOKEN, "flux");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_FLUX_H
