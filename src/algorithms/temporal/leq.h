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

#ifndef ESSENTIA_LEQ_H
#define ESSENTIA_LEQ_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Leq : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _leq;

 public:
  Leq() {
    declareInput(_signal, "signal", "the input signal (must be non-empty)");
    declareOutput(_leq, "leq", "the equivalent sound level estimate");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "accumulatoralgorithm.h"

namespace essentia {
namespace streaming {

class Leq : public AccumulatorAlgorithm {

 protected:
  Sink<Real> _signal;
  Source<Real> _leq;

  Real _energy;
  int _size;

 public:
  Leq() {
    declareInputStream(_signal, "signal", "the input signal (must be non-empty)");
    declareOutputResult(_leq, "leq", "the equivalent sound level estimate");
    reset();
  }

  void reset();
  void declareParameters() {}

  void consume();
  void finalProduce();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_LEQ_H
