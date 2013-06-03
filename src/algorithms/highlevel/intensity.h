/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_INTENSITY_H
#define ESSENTIA_INTENSITY_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class Intensity : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<int> _intensity;

  // spectrum
  Algorithm* _frameCutter;
  Algorithm* _windowing;
  Algorithm* _spectrum;

  // spectral complexity
  Algorithm* _spectralComplexity;

  // spectral kurtosis
  Algorithm* _centralMoments;
  Algorithm* _distributionShape;

  // spectral rollOff
  Algorithm* _rollOff;

  // signal dissonance
  Algorithm* _spectralPeaks;
  Algorithm* _dissonance;

 public:
  Intensity() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_intensity, "intensity", "the intensity value");

    _frameCutter = AlgorithmFactory::create("FrameCutter");
    _windowing = AlgorithmFactory::create("Windowing");
    _spectrum = AlgorithmFactory::create("Spectrum");
    _spectralComplexity = AlgorithmFactory::create("SpectralComplexity");
    _centralMoments = AlgorithmFactory::create("CentralMoments");
    _distributionShape = AlgorithmFactory::create("DistributionShape");
    _rollOff = AlgorithmFactory::create("RollOff");
    _spectralPeaks = AlgorithmFactory::create("SpectralPeaks");
    _dissonance = AlgorithmFactory::create("Dissonance");
  }

  ~Intensity() {
    delete _frameCutter;
    delete _windowing;
    delete _spectrum;
    delete _spectralComplexity;
    delete _centralMoments;
    delete _distributionShape;
    delete _rollOff;
    delete _spectralPeaks;
    delete _dissonance;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the input audio sampling rate [Hz]", "(0,inf)", 44100.);
  }

  void reset() {
    _frameCutter->reset();
    _windowing->reset();
    _spectrum->reset();
    _spectralComplexity->reset();
    _centralMoments->reset();
    _distributionShape->reset();
    _rollOff->reset();
    _spectralPeaks->reset();
    _dissonance->reset();
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_INTENSITY_H
