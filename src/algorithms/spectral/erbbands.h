/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_ERBBANDS_H
#define ESSENTIA_ERBBANDS_H

#include "essentiamath.h"
#include "algorithm.h"
#include <complex>

namespace essentia {
namespace standard {

class ERBBands : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrumInput;
  Output<std::vector<Real> > _bandsOutput;

 public:
  ERBBands() {
    declareInput(_spectrumInput, "spectrum", "the audio spectrum");
    declareOutput(_bandsOutput, "bands", "the magnitudes of each band");
  }

  void declareParameters() {
    declareParameter("inputSize", "the size of the spectrum", "(1,inf)", 513);
    declareParameter("numberBands", "the number of output bands", "(1,inf)", 40);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("lowFrequencyBound", "a lower-bound limit for the frequencies to be included in the bands", "[0,inf)", 50.0);
    declareParameter("highFrequencyBound", "an upper-bound limit for the frequencies to be included in the bands", "[0,inf)", 22050.0);
    declareParameter("width", "filter width with respect to ERB", "(0,inf)", 1.0);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* version;
  static const char* description;

 protected:

  void createFilters(int spectrumSize);
  void calculateFilterFrequencies();

  std::vector<std::vector<Real> > _filterCoefficients;
  std::vector<Real> _filterFrequencies;
  int _numberBands;

  Real _sampleRate;
  Real _maxFrequency;
  Real _minFrequency;
  Real _width;

  static const Real EarQ;
  static const Real minBW;

};

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {


class ERBBands : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrumInput;
  Source<std::vector<Real> > _bandsOutput;

 public:
  ERBBands() {
    declareAlgorithm("ERBBands");
    declareInput(_spectrumInput, TOKEN, "spectrum");
    declareOutput(_bandsOutput, TOKEN, "bands");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_ERBBands_H
