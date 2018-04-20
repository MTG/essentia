/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_BESSELFUNCION_H
#define ESSENTIA_BESSELFUNCION_H

#include "algorithm.h"
#include <tr1/cmath>

namespace essentia {
namespace standard {

class BesselFunction : public Algorithm {


 protected:
  int _v;
  Input<std::vector<Real> > _x;
  Output<std::vector<Real> > _y;

 public:
  BesselFunction() {
    declareInput(_x, "x", "the input vector");
    declareOutput(_y, "y", "the value of Bessel function");
  }

  ~BesselFunction() {}

  void declareParameters() {
    declareParameter("v", "the order of the Bessel Function", "[0,inf)", 0);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace essentia
} // namespace standard

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class BesselFunction : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _x;
  Source<std::vector<Real> > _y;

  int _v;

 public:
  BesselFunction() {
    declareAlgorithm("BesselFunction");
    declareInput(_x, TOKEN, "x");
    declareOutput(_y, TOKEN, "y");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BESSELFUNCION_H
