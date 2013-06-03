/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_SPECTRUM_H
#define ESSENTIA_SPECTRUM_H

#include "algorithmfactory.h"
#include <complex>

namespace essentia {
namespace standard {

class Spectrum : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _spectrum;

  Algorithm* _fft;
  Algorithm* _magnitude;
  std::vector<std::complex<Real> > _fftBuffer;

 public:
  Spectrum() {
    declareInput(_signal, "frame", "the input audio frame");
    declareOutput(_spectrum, "spectrum", "the magnitude spectrum of the input audio signal");

    _fft = AlgorithmFactory::create("FFT");
    _magnitude = AlgorithmFactory::create("Magnitude");
  }

  ~Spectrum() {
    delete _fft;
    delete _magnitude;
  }

  void declareParameters() {
    declareParameter("size", "the expected size of the input audio signal (this is an optional parameter to optimize memory allocation)", "[1,inf)", 1024);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Spectrum : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<std::vector<Real> > _spectrum;

 public:
  Spectrum() {
    declareAlgorithm("Spectrum");
    declareInput(_signal, TOKEN, "frame");
    declareOutput(_spectrum, TOKEN, "spectrum");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SPECTRUM_H
