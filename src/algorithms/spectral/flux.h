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

#ifndef ESSENTIA_FLUX_H
#define ESSENTIA_FLUX_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Flux : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _flux;

  std::vector<Real> _spectrumMemory;
  std::string _norm;
  bool _halfRectify;

 public:
  Flux() {
    declareInput(_spectrum, "spectrum", "the input spectrum");
    declareOutput(_flux, "flux", "the spectral flux of the input spectrum");
  }

  void declareParameters() {
    declareParameter("norm", "the norm to use for difference computation", "{L1,L2}", "L2");
    declareParameter("halfRectify", "half-rectify the differences in each spectrum bin", "{true,false}", false);
  }

  void configure();
  void compute();

  void reset() {
    _spectrumMemory.clear();
  }

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Flux : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _flux;

 public:
  Flux() {
    declareAlgorithm("Flux");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_flux, TOKEN, "flux");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_FLUX_H
