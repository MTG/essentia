/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_LOUDNESSVICKERS_H
#define ESSENTIA_LOUDNESSVICKERS_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class LoudnessVickers : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _loudness;

  Real _sampleRate;
  Real _Vms;
  Real _c;
  Algorithm* _filtering;

 public:
  LoudnessVickers() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_loudness, "loudness", "the Vickers loudness [dB]");

    _filtering = AlgorithmFactory::create("IIR");
  }

  ~LoudnessVickers() {
    if (_filtering) delete _filtering;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the audio sampling rate of the input signal which is used to create the weight vector [Hz] (currently, this algorithm only works on signals with a sampling rate of 44100Hz)", "[44100,44100]", 44100.);
  }

  void configure();
  void compute();

  void reset() {
    _filtering->reset();
    _Vms = 0.0;
  }

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class LoudnessVickers : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _loudness;

 public:
  LoudnessVickers() {
    declareAlgorithm("LoudnessVickers");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_loudness, TOKEN, "loudness");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_LOUDNESSVICKERS_H
