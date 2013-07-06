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

#ifndef ESSENTIA_POLAR2CARTESIAN_H
#define ESSENTIA_POLAR2CARTESIAN_H

#include "algorithm.h"
#include "streamingalgorithmwrapper.h"
#include <complex>

namespace essentia {
namespace standard {

class PolarToCartesian : public Algorithm {

 private:
  Input<std::vector<Real> > _magnitude;
  Input<std::vector<Real> > _phase;
  Output<std::vector<std::complex<Real> > > _complex;

 public:
  PolarToCartesian() {
    declareInput(_magnitude, "magnitude", "the magnitude vector");
    declareInput(_phase, "phase", "the phase vector");
    declareOutput(_complex, "complex", "the resulting complex vector");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
namespace streaming {

class PolarToCartesian : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _magnitude;
  Sink<std::vector<Real> > _phase;
  Source<std::vector<std::complex<Real> > > _complex;

 public:
  PolarToCartesian() {
    declareAlgorithm("PolarToCartesian");
    declareInput(_magnitude, TOKEN, "magnitude");
    declareInput(_phase, TOKEN, "phase");
    declareOutput(_complex, TOKEN, "complex");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_POLAR2CARTESIAN_H
