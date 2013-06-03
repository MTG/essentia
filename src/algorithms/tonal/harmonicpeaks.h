/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_HARMONICPEAKS_H
#define ESSENTIA_HARMONICPEAKS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class HarmonicPeaks : public Algorithm {

 protected:
  Input<std::vector<Real> > _frequencies;
  Input<std::vector<Real> > _magnitudes;
  Input<Real> _pitch;
  Output<std::vector<Real> > _harmonicFrequencies;
  Output<std::vector<Real> > _harmonicMagnitudes;

 public:
  HarmonicPeaks() {
    declareInput(_frequencies, "frequencies", "the frequencies of the spectral peaks [Hz] (ascending order)");
    declareInput(_magnitudes, "magnitudes", "the magnitudes of the spectral peaks (ascending frequency order)");
    declareInput(_pitch, "pitch", "an estimate of the fundamental frequency of the signal [Hz]");
    declareOutput(_harmonicFrequencies, "harmonicFrequencies", "the frequencies of harmonic peaks [Hz]");
    declareOutput(_harmonicMagnitudes, "harmonicMagnitudes", "the magnitudes of harmonic peaks");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class HarmonicPeaks : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _magnitudes;
  Sink<Real> _pitch;
  Source<std::vector<Real> > _harmonicFrequencies;
  Source<std::vector<Real> > _harmonicMagnitudes;

 public:
  HarmonicPeaks() {
    declareAlgorithm("HarmonicPeaks");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareInput(_pitch, TOKEN, "pitch");
    declareOutput(_harmonicFrequencies, TOKEN, "harmonicFrequencies");
    declareOutput(_harmonicMagnitudes, TOKEN, "harmonicMagnitudes");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_HARMONICPEAKS_H
