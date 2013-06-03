/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_ONSETRATE_H
#define ESSENTIA_ONSETRATE_H

#include "algorithmfactory.h"
#include "network.h"

namespace essentia {
namespace standard {

class OnsetRate : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _onsetTimes;
  Output<Real> _onsetRate;

  Real _sampleRate;
  int _frameSize;
  int _hopSize;
  Real _frameRate;
  int _zeroPadding;

  // Pre-processing
  Algorithm* _frameCutter;
  Algorithm* _windowing;

  // FFT
  Algorithm* _fft;
  Algorithm* _cartesian2polar;

  // Onsets
  Algorithm* _onsetHfc;
  Algorithm* _onsetComplex;
  Algorithm* _onsets;

public:
  OnsetRate() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_onsetTimes, "onsets", "the positions of detected onsets [s]");
    declareOutput(_onsetRate, "onsetRate", "the number of onsets per second");

    // Pre-processing
    _frameCutter = AlgorithmFactory::create("FrameCutter");
    _windowing = AlgorithmFactory::create("Windowing");

    // FFT
    _fft = AlgorithmFactory::create("FFT");
    _cartesian2polar = AlgorithmFactory::create("CartesianToPolar");

    // Onsets
    _onsetHfc = AlgorithmFactory::create("OnsetDetection");
    _onsetComplex = AlgorithmFactory::create("OnsetDetection");
    _onsets = AlgorithmFactory::create("Onsets");
  }

  ~OnsetRate();

  void declareParameters() {}

  void compute();
  void configure();

  void reset() {
    _frameCutter->reset();
    _onsets->reset();
    _onsetComplex->reset();
  }

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia


#include "pool.h"
#include "streamingalgorithmcomposite.h"

namespace essentia {
namespace streaming {

class OnsetRate : public AlgorithmComposite {

 protected:
  SinkProxy<Real> _signal;

  Source<std::vector<Real> > _onsetTimes;
  Source<Real> _onsetRate;

  Algorithm* _frameCutter;
  Algorithm* _windowing;
  Algorithm* _fft;
  Algorithm* _cart2polar;
  Algorithm* _onsetHfc;
  Algorithm* _onsetComplex;
  standard::Algorithm* _onsets;

  scheduler::Network* _network;

  Pool _pool;

  Real _sampleRate;
  int _frameSize;
  int _hopSize;
  Real _frameRate;
  int _zeroPadding;

  int _preferredBufferSize;

 public:
  OnsetRate();
  ~OnsetRate();

  void declareParameters() {};

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
    declareProcessStep(SingleShot(this));
  }

  void configure();
  AlgorithmStatus process();
  void reset();
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_ONSETRATE_H
