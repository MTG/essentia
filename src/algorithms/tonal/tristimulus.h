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

#ifndef ESSENTIA_TRISTIMULUS_H
#define ESSENTIA_TRISTIMULUS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Tristimulus : public Algorithm {

 protected:
  Input<std::vector<Real> > _frequencies;
  Input<std::vector<Real> > _magnitudes;
  Output<std::vector<Real> > _tristimulus;

 public:
  Tristimulus() {
    declareInput(_frequencies, "frequencies", "the frequencies of the harmonic peaks ordered by frequency");
    declareInput(_magnitudes, "magnitudes", "the magnitudes of the harmonic peaks ordered by frequency");
    declareOutput(_tristimulus, "tristimulus", "a three-element vector that measures the mixture of harmonics of the given spectrum");
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

class Tristimulus : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _magnitudes;
  Source<std::vector<Real> > _tristimulus;

 public:
  Tristimulus() {
    declareAlgorithm("Tristimulus");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_tristimulus, TOKEN, "tristimulus");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_TRISTIMULUS_H
