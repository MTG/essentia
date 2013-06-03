/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_ENERGYBANDRATIO_H
#define ESSENTIA_ENERGYBANDRATIO_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class EnergyBandRatio : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _energyBandRatio;

  Real _startFreqNormalized, _stopFreqNormalized;

 public:
  EnergyBandRatio() {
    declareInput(_spectrum, "spectrum", "the input audio spectrum");
    declareOutput(_energyBandRatio, "energyBandRatio", "the energy ratio of the specified band over the total energy");
  }

  void declareParameters() {
    declareParameter("startFrequency", "the frequency from which to start summing the energy [Hz]", "[0,inf)", 0.0);
    declareParameter("stopFrequency", "the frequency up to which to sum the energy [Hz]", "[0,inf)", 100.0);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
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

class EnergyBandRatio : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _energyBandRatio;

 public:
  EnergyBandRatio() {
    declareAlgorithm("EnergyBandRatio");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_energyBandRatio, TOKEN, "energyBandRatio");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_ENERGYBANDRATIO_H
