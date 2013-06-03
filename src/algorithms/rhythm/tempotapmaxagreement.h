/*
 * Copyright (C) 2006-2012 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_TEMPOTAPMAXAGREEMENT_H
#define ESSENTIA_TEMPOTAPMAXAGREEMENT_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class TempoTapMaxAgreement : public Algorithm {

 protected:
  Input<std::vector<std::vector<Real> > > _tickCandidates;
  Output<std::vector<Real> > _ticks;

 public:
  TempoTapMaxAgreement() {
    declareInput(_tickCandidates, "tickCandidates", "the tick candidates estimated using different beat trackers (or features) [s]");
    declareOutput(_ticks, "ticks", "the list of resulting ticks [s]");
  }

  ~TempoTapMaxAgreement() {
  };

  void declareParameters() {
  }

  void reset();
  void configure();
  void compute();

  static const char* name;
  static const char* description;

 private:
  static const Real _minTickTime = 5.;  // ignore peaks before this time [s]
  static const int _numberBins = 40; // number of histogram bins for information gain method
 
  std::vector<Real> _histogramBins;
  std::vector<Real> _binValues;

  // parameters for the continuity-based method
  static const Real _phaseThreshold = 0.175; // size of tolerance window for beat phase 
  static const Real _periodThreshold = 0.175; // size of tolerance window for beat period 

  Real computeBeatInfogain(std::vector<Real>& ticks1, std::vector<Real>& ticks2);

  void removeFirstSeconds(std::vector<Real>& ticks);
  void FindBeatError(const std::vector<Real>& ticks1,
                     const std::vector<Real>& ticks2,
                     std::vector<Real>& beatError);
  Real FindEntropy(std::vector<Real>& beatError);
  size_t closestTick(const std::vector<Real>& ticks, Real x);
  void histogram(const std::vector<Real>& array, std::vector<Real>& counter);

}; // class TempoTapMaxAgreement

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class TempoTapMaxAgreement : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::vector<Real> > > _tickCandidates;
  Source<std::vector<Real> > _ticks;

 public:
  TempoTapMaxAgreement() {
    declareAlgorithm("TempoTapMaxAgreement");
    declareInput(_tickCandidates, TOKEN, "tickCandidates");
    declareOutput(_ticks, TOKEN, "ticks");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_TEMPOTAPMAXAGREEMENT_H
