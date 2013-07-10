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

#ifndef DERIVATIVE_H
#define DERIVATIVE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Derivative : public Algorithm {

 private:
  Input<std::vector<Real> > _input;
  Output<std::vector<Real> > _output;

 public:
  Derivative() {
    declareInput(_input, "signal", "the input signal");
    declareOutput(_output, "signal", "the derivative of the input signal");
  }
  ~Derivative() {}
  void declareParameters() {}
  void compute();
  void configure() {}

  static const char* name;
  static const char* description;
};

}// namespace standard
}// namespace essentia

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {


class Derivative : public Algorithm {

 protected:
  Sink<Real> _input;
  Source<Real> _output;
  Real _oldValue;

 public:
  Derivative() {
    declareInput(_input, 1, "signal", "the input signal");
    declareOutput(_output, 1, "signal", "the derivative of the input signal");
  }

  ~Derivative() {}

  void reset();
  void declareParameters() {}
  void configure();
  AlgorithmStatus process();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // DERIVATIVE_H
