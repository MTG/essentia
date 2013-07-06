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

#ifndef ESSENTIA_CENTROID_H
#define ESSENTIA_CENTROID_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Centroid : public Algorithm {

 protected:
  Input<std::vector<Real> > _array;
  Output<Real> _centroid;

  Real _range;

 public:
  Centroid() {
    declareInput(_array, "array", "the input array");
    declareOutput(_centroid, "centroid", "the centroid of the array");
  }

  void declareParameters() {
    declareParameter("range", "the range of the input array, used for normalizing the results", "(0,inf)", 1.0);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Centroid : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _centroid;

 public:
  Centroid() {
    declareAlgorithm("Centroid");
    declareInput(_array, TOKEN, "array");
    declareOutput(_centroid, TOKEN, "centroid");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_CENTROID_H
