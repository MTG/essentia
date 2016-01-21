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
