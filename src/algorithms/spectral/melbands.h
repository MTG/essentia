/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_MELBANDS_H
#define ESSENTIA_MELBANDS_H

#include "essentiamath.h"
#include "algorithm.h"

namespace essentia {
namespace standard {

class MelBands : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrumInput;
  Output<std::vector<Real> > _bandsOutput;

 public:
  MelBands() {
    declareInput(_spectrumInput, "spectrum", "the audio spectrum");
    declareOutput(_bandsOutput, "bands", "the energy in mel bands");
  }

  void declareParameters() {
    declareParameter("inputSize", "the size of the spectrum", "(1,inf)", 1025);
    declareParameter("numberBands", "the number of output bands", "(1,inf)", 24);
    declareParameter("sampleRate", "the sample rate", "(0,inf)", 44100.);
    declareParameter("lowFrequencyBound", "a lower-bound limit for the frequencies to be included in the bands", "[0,inf)", 0.0);
    declareParameter("highFrequencyBound", "an upper-bound limit for the frequencies to be included in the bands", "[0,inf)", 22050.0);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

 protected:

  void createFilters(int spectrumSize);
  void calculateFilterFrequencies();

  std::vector<std::vector<Real> > _filterCoefficients;
  std::vector<Real> _filterFrequencies;
  int _numBands;
  Real _sampleRate;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class MelBands : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrumInput;
  Source<std::vector<Real> > _bandsOutput;

 public:
  MelBands() {
    declareAlgorithm("MelBands");
    declareInput(_spectrumInput, TOKEN, "spectrum");
    declareOutput(_bandsOutput, TOKEN, "bands");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_MELBANDS_H
