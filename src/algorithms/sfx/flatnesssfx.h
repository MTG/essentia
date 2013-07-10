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

#ifndef ESSENTIA_FLATNESSSFX_H
#define ESSENTIA_FLATNESSSFX_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class FlatnessSFX : public Algorithm {

 protected:
  Input<std::vector<Real> > _envelope;
  Output<Real> _flatnessSFX;

 public:
  FlatnessSFX() {
    declareInput(_envelope, "envelope", "the envelope of the signal");
    declareOutput(_flatnessSFX, "flatness", "the flatness coefficient");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;

  // these thresholds are given in percentage of the total signal length
  // they are used to determine the values that are at the lower threshold (5%)
  // and the upper threshold (80%) respectively
  static const Real lowerThreshold;
  static const Real upperThreshold;

 private:
  Real rollOff(const std::vector<Real>& envelope, Real x) const;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class FlatnessSFX : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _envelope;
  Source<Real> _flatnessSFX;

 public:
  FlatnessSFX() {
    declareAlgorithm("FlatnessSFX");
    declareInput(_envelope, TOKEN, "envelope");
    declareOutput(_flatnessSFX, TOKEN, "flatness");
  }

};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_FLATNESSSFX_H
