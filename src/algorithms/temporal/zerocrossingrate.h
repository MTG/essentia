/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_ZEROCROSSINGRATE_H
#define ESSENTIA_ZEROCROSSINGRATE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class ZeroCrossingRate : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _zeroCrossingRate;
  float _threshold;

 public:
  ZeroCrossingRate() : _threshold(0) {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_zeroCrossingRate, "zeroCrossingRate", "the zero-crossing rate");
  }

  void declareParameters() {
    declareParameter("threshold", "the threshold which will be taken as the zero axis in both positive and negative sign", "[0,inf]", 0.0);
  }
  void compute();
  void configure();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class ZeroCrossingRate : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _zeroCrossingRate;

 public:
  ZeroCrossingRate() {
    declareAlgorithm("ZeroCrossingRate");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_zeroCrossingRate, TOKEN, "zeroCrossingRate");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_ZEROCROSSINGRATE_H
