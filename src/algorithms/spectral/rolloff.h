/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_ROLLOFF_H
#define ESSENTIA_ROLLOFF_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class RollOff : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _rolloff;

 public:
  RollOff() {
    declareInput(_spectrum, "spectrum", "the input audio spectrum (must have more than one elements)");
    declareOutput(_rolloff, "rollOff", "the roll-off frequency [Hz]");
  }

  void declareParameters() {
    declareParameter("cutoff", "the ratio of total energy to attain before yielding the roll-off frequency", "(0,1)", 0.85);
    declareParameter("sampleRate", "the sampling rate of the audio signal (used to normalize rollOff) [Hz]", "(0,inf)", 44100.);
  }
  void compute();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class RollOff : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _rolloff;

 public:
  RollOff() {
    declareAlgorithm("RollOff");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_rolloff, TOKEN, "rollOff");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_ROLLOFF_H
