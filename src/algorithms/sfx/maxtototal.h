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

#ifndef ESSENTIA_MAXTOTOTAL_H
#define ESSENTIA_MAXTOTOTAL_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class MaxToTotal : public Algorithm {

 protected:
  Input<std::vector<Real> > _envelope;
  Output<Real> _maxToTotal;

 public:
  MaxToTotal() {
    declareInput(_envelope, "envelope", "the envelope of the signal");
    declareOutput(_maxToTotal, "maxToTotal", "the maximum amplitude position to total length ratio");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;
};

} // namespace essentia
} // namespace standard

#include "accumulatoralgorithm.h"

namespace essentia {
namespace streaming {

class MaxToTotal : public AccumulatorAlgorithm {

 protected:
  Sink<Real> _envelope;
  Source<Real> _maxToTotal;

  int _size, _maxIdx;
  Real _max;

 public:
  MaxToTotal() {
    declareInputStream(_envelope, "envelope", "the envelope of the signal");
    declareOutputResult(_maxToTotal, "maxToTotal", "the maximum amplitude position to total length ratio");
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

#endif // ESSENTIA_MAXTOTOTAL_H
