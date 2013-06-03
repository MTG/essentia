/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_ONSETDETECTION_H
#define ESSENTIA_ONSETDETECTION_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class OnsetDetection : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Input<std::vector<Real> > _phase;
  Output<Real> _onsetDetection;

  Algorithm* _hfc;
  Algorithm* _flux;
  Algorithm* _melBands;
  std::string _method;

 public:
  OnsetDetection() {
    declareInput(_spectrum, "spectrum", "the input spectrum");
    declareInput(_phase, "phase", "the phase vector corresponding to this spectrum--used only by the \"complex\" method");
    declareOutput(_onsetDetection, "onsetDetection", "the value of the detection function in the current frame");

    _hfc = AlgorithmFactory::create("HFC");
    _flux = AlgorithmFactory::create("Flux");
    _melBands = AlgorithmFactory::create("MelBands");
  }

  ~OnsetDetection() {
    if (_hfc) delete _hfc;
    if (_flux) delete _flux;
    if (_melBands) delete _melBands;
  }

  void declareParameters() {
    declareParameter("method", "the method used for onset detection", "{hfc,complex,complex_phase,flux,melflux,rms}", "hfc");
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.0);
  }

  void reset();
  void configure();
  void compute();

  static const char* name;
  static const char* description;

  std::vector<Real> _phase_1;
  std::vector<Real> _phase_2;
  std::vector<Real> _spectrum_1;
  Real _rmsOld;
  bool _firstFrame;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class OnsetDetection : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _phase;
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _onsetDetection;

 public:
  OnsetDetection() {
    declareAlgorithm("OnsetDetection");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareInput(_phase, TOKEN, "phase");
    declareOutput(_onsetDetection, TOKEN, "onsetDetection");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_ONSETDETECTION_H
