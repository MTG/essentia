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
