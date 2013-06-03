/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_DYNAMICCOMPLEXITY_H
#define ESSENTIA_DYNAMICCOMPLEXITY_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class DynamicComplexity : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _complexity;
  Output<Real> _loudness;

  int _frameSize;
  Real _sampleRate;

 public:
  DynamicComplexity() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_complexity, "dynamicComplexity", "the dynamic complexity coefficient");
    declareOutput(_loudness, "loudness", "an estimate of the loudness [dB]");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the frame size [s]", "(0,inf)", 0.2);
  }


  void configure();
  void compute();

  static const char* name;
  static const char* description;

 protected:
  void filter(std::vector<Real>& result, const std::vector<Real>& input) const;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmcomposite.h"
#include "pool.h"

namespace essentia {
namespace streaming {

class DynamicComplexity : public AlgorithmComposite {

 protected:
  SinkProxy<Real> _signal;
  Source<Real> _complexity;
  Source<Real> _loudness;

  Pool _pool;
  Algorithm* _poolStorage;
  standard::Algorithm* _dynAlgo;

 public:
  DynamicComplexity();
  ~DynamicComplexity() {
    delete _poolStorage;
    delete _dynAlgo;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the frame size [s]", "(0,inf)", 0.2);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_poolStorage));
    declareProcessStep(SingleShot(this));
  }

  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_DYNAMICCOMPLEXITY_H
