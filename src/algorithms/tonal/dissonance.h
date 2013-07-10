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

#ifndef ESSENTIA_DISSONANCE_H
#define ESSENTIA_DISSONANCE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Dissonance : public Algorithm {

 protected:
  Input<std::vector<Real> > _frequencies;
  Input<std::vector<Real> > _magnitudes;
  Output<Real> _dissonance;

 public:
  Dissonance() {
    declareInput(_frequencies, "frequencies", "the frequencies of the spectral peaks (must be sorted by frequency)");
    declareInput(_magnitudes, "magnitudes", "the magnitudes of the spectral peaks (must be sorted by frequency");
    declareOutput(_dissonance, "dissonance", "the dissonance of the audio signal (0 meaning completely consonant, and 1 meaning completely dissonant)");
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

class Dissonance : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _magnitudes;
  Source<Real> _dissonance;

 public:
  Dissonance() {
    declareAlgorithm("Dissonance");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_dissonance, TOKEN, "dissonance");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_DISSONANCE_H
