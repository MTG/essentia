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

#ifndef ESSENTIA_STRONGPEAK_H
#define ESSENTIA_STRONGPEAK_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class StrongPeak : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _strongPeak;

 public:
  StrongPeak() {
    declareInput(_spectrum, "spectrum", "the input spectrum (must be greater than one element and cannot contain negative values)");
    declareOutput(_strongPeak, "strongPeak", "the Strong Peak ratio");
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

class StrongPeak : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _strongPeak;

 public:
  StrongPeak() {
    declareAlgorithm("StrongPeak");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_strongPeak, TOKEN, "strongPeak");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_STRONGPEAK_H
