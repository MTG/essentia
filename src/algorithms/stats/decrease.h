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

#ifndef ESSENTIA_DECREASE_H
#define ESSENTIA_DECREASE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Decrease : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<Real> _decrease;

 public:
  Decrease() {
    declareInput(_array, "array", "the input array");
    declareOutput(_decrease, "decrease", "the decrease of the input array");
  }

  void declareParameters() {
    declareParameter("range", "the range of the input array, used for normalizing the results", "(-inf,inf)", 1.0);
  }

  void configure();

  void compute();

  static const char* name;
  static const char* description;

 protected:
  Real _range;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Decrease : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _decrease;

 public:
  Decrease() {
    declareAlgorithm("Decrease");
    declareInput(_array, TOKEN, "array");
    declareOutput(_decrease, TOKEN, "decrease");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_DECREASE_H
