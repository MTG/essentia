/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
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
  Output<Real> _confidence;

 public:
  TempoTapMaxAgreement() {
    declareInput(_tickCandidates, "tickCandidates", "the tick candidates estimated using different beat trackers (or features) [s]");
    declareOutput(_ticks, "ticks", "the list of resulting ticks [s]");
    declareOutput(_confidence, "confidence", "confidence with which the ticks were detected [0, 5.32]");
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
                                     // corresponds to Log2(40) = 5.32 maximum
                                     // confidence value

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
  Source<Real> _confidence;

 public:
  TempoTapMaxAgreement() {
    declareAlgorithm("TempoTapMaxAgreement");
    declareInput(_tickCandidates, TOKEN, "tickCandidates");
    declareOutput(_ticks, TOKEN, "ticks");
    declareOutput(_confidence, TOKEN, "confidence");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_TEMPOTAPMAXAGREEMENT_H
