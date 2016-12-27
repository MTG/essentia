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

#ifndef ESSENTIA_CENTRALMOMENTS_H
#define ESSENTIA_CENTRALMOMENTS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class CentralMoments : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<std::vector<Real> > _centralMoments;

 protected:
  Real _range;
  std::string _mode;

 public:
  CentralMoments() {
    declareInput(_array, "array", "the input array");
    declareOutput(_centralMoments, "centralMoments", "the central moments of the input array");
  }

  void declareParameters() {
    declareParameter("mode", "compute central moments considering array values as a probability density function over array index or as sample points of a distribution", "{pdf,sample}", "pdf");
    declareParameter("range", "the range of the input array, used for normalizing the results in the 'pdf' mode", "(0,inf)", 1.0);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class CentralMoments : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<std::vector<Real> > _centralMoments;

 public:
  CentralMoments() {
    declareAlgorithm("CentralMoments");
    declareInput(_array, TOKEN, "array");
    declareOutput(_centralMoments, TOKEN, "centralMoments");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_CENTRALMOMENTS_H
