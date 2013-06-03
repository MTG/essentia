/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_HARMONIC_BPM_H
#define ESSENTIA_HARMONIC_BPM_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class HarmonicBpm : public Algorithm {

 private:
  Input<std::vector<Real> > _bpmCandidates;
  Output<std::vector<Real> > _harmonicBpms;

  Real _threshold;
  Real _bpm;
  Real _tolerance;

 public:
  HarmonicBpm() {
    declareInput(_bpmCandidates, "bpms", "list of bpm candidates");
    declareOutput(_harmonicBpms, "harmonicBpms", "a list of bpms which are harmonically related to the bpm parameter ");
  }

  ~HarmonicBpm() {}

  void declareParameters() {
    declareParameter("bpm", "the bpm used to find its harmonics", "[1,inf)", 60);
    declareParameter("threshold", "bpm threshold below which greatest common divisors are discarded", "[1,inf)", 20.0);
    declareParameter("tolerance", "percentage tolerance to consider two bpms are equal or equal to a harmonic", "[0,inf)", 5.0);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* version;
  static const char* description;

 private:
  std::vector<Real> findHarmonicBpms(const std::vector<Real>& bpms);
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class HarmonicBpm : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _bpmCandidates;
  Source<std::vector<Real> > _harmonicBpms;

 public:
  HarmonicBpm() {
    declareAlgorithm("HarmonicBpm");
    declareInput(_bpmCandidates, TOKEN, "bpms");
    declareOutput(_harmonicBpms, TOKEN, "harmonicBpms");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_HARMONIC_BPM_H
