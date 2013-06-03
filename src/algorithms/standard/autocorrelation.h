/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_AUTOCORRELATION_H
#define ESSENTIA_AUTOCORRELATION_H

#include "algorithmfactory.h"
#include <complex>

namespace essentia {
namespace standard {

class AutoCorrelation : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _correlation;
  bool _unbiasedNormalization;
  std::vector<std::complex<Real> > _fftBuffer;
  std::vector<Real> _corr;
  std::vector<Real> _paddedSignal;

  Algorithm* _fft;
  Algorithm* _ifft;

 public:
  AutoCorrelation() : _fftBuffer(0), _corr(0), _paddedSignal(0) {
    declareInput(_signal, "array", "the array to be analyzed");
    declareOutput(_correlation, "autoCorrelation", "the autocorrelation vector");

    _fft = AlgorithmFactory::create("FFT");
    _ifft = AlgorithmFactory::create("IFFT");
  }

  ~AutoCorrelation() {
    delete _fft;
    delete _ifft;
  }

  void declareParameters() {
    declareParameter("normalization", "type of normalization to compute: either 'standard' (default) or 'unbiased'", "{standard,unbiased}", "standard");
  }

  void configure();
  void compute();

 protected:
  void createFFTObject(int size);

 public:
  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class AutoCorrelation : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<std::vector<Real> > _correlation;

 public:
  AutoCorrelation() {
    declareAlgorithm("AutoCorrelation");
    declareInput(_signal, TOKEN, "array");
    declareOutput(_correlation, TOKEN, "autoCorrelation");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_AUTOCORRELATION_H
