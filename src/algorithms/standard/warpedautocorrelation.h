/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_WARPEDAUTOCORRELATION_H
#define ESSENTIA_WARPEDAUTOCORRELATION_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class WarpedAutoCorrelation : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _warpedAutoCorrelation;

 public:
  WarpedAutoCorrelation() {
    declareInput(_signal, "array", "the array to be analyzed");
    declareOutput(_warpedAutoCorrelation, "warpedAutoCorrelation", "the warped auto-correlation vector");
  }

  void declareParameters() {
    declareParameter("maxLag", "the maximum lag for which the auto-correlation is computed (inclusive) (must be smaller than signal size) ", "(0,inf)", 1);
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

 private:
  Real _lambda;
  std::vector<Real> _tmp;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class WarpedAutoCorrelation : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<std::vector<Real> > _warpedAutoCorrelation;

 public:
  WarpedAutoCorrelation() {
    declareAlgorithm("WarpedAutoCorrelation");
    declareInput(_signal, TOKEN, "array");
    declareOutput(_warpedAutoCorrelation, TOKEN, "warpedAutoCorrelation");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_WARPEDAUTOCORRELATION_H
