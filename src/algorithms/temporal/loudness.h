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

#ifndef ESSENTIA_LOUDNESS_H
#define ESSENTIA_LOUDNESS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Loudness : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _loudness;

 public:
  Loudness() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_loudness, "loudness", "the loudness of the input signal");
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

class Loudness : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _loudness;

 public:
  Loudness() {
    declareAlgorithm("Loudness");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_loudness, TOKEN, "loudness");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_LOUDNESS_H
