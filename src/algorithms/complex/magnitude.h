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


#ifndef ESSENTIA_MAGNITUDE_H
#define ESSENTIA_MAGNITUDE_H


#include "algorithm.h"
#include "streamingalgorithmwrapper.h"
#include <complex>


namespace essentia {
namespace standard {


class Magnitude : public Algorithm {

 private:
  Input<std::vector<std::complex<Real> > > _complex;
  Output<std::vector<Real> > _magnitude;

 public:
  Magnitude() {
    declareInput(_complex, "complex", "the input vector of complex numbers");
    declareOutput(_magnitude, "magnitude", "the magnitudes of the input vector");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;
};


} // namespace standard
namespace streaming {


class Magnitude : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::complex<Real> > > _complex;
  Source<std::vector<Real> > _magnitude;

 public:
  Magnitude() {
    declareAlgorithm("Magnitude");
    declareInput(_complex, TOKEN, "complex");
    declareOutput(_magnitude, TOKEN, "magnitude");
  }
};


} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_MAGNITUDE_H
