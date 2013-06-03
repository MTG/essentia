/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_FREQBANDS_H
#define ESSENTIA_FREQBANDS_H

#include "algorithm.h"
#include "essentiautil.h"

namespace essentia {
namespace standard {

class FrequencyBands : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrumInput;
  Output<std::vector<Real> > _bandsOutput;

 public:
  FrequencyBands() {
    declareInput(_spectrumInput, "spectrum", "the input spectrum (must be greater than size one)");
    declareOutput(_bandsOutput, "bands", "the energy in each band");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);

    Real freqBands[] = {0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0,
                        920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0,
                        3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0,
                        15500.0, 20500.0, 27000.0};
    declareParameter("frequencyBands", "list of frequency ranges in to which the spectrum is divided (these must be in ascending order and connot contain duplicates)", "", arrayToVector<Real>(freqBands));
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

 protected:
  std::vector<Real> _bandFrequencies;
  Real _sampleRate;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class FrequencyBands : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrumInput;
  Source<std::vector<Real> > _bandsOutput;

 public:
  FrequencyBands() {
    declareAlgorithm("FrequencyBands");
    declareInput(_spectrumInput, TOKEN, "spectrum");
    declareOutput(_bandsOutput, TOKEN, "bands");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BARKBANDS_H
