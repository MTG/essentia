/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_ENERGYBAND_H
#define ESSENTIA_ENERGYBAND_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class EnergyBand : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _energyBand;

  Real _normStartIdx, _normStopIdx;

 public:
  EnergyBand() {
    declareInput(_spectrum, "spectrum", "the input frequency spectrum");
    declareOutput(_energyBand, "energyBand", "the energy in the frequency band");
  }

  void declareParameters() {
    declareParameter("startCutoffFrequency", "the start frequency from which to sum the energy [Hz]", "[0,inf)", 0.0);
    declareParameter("stopCutoffFrequency", "the stop frequency to which to sum the energy [Hz]", "(0,inf)", 100.0);
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace essentia
} // namespace standard

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class EnergyBand : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _energyBand;

 public:
  EnergyBand() {
    declareAlgorithm("EnergyBand");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_energyBand, TOKEN, "energyBand");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_ENERGYBAND_H
