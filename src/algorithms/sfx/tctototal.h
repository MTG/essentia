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

#ifndef ESSENTIA_TCTOTOTAL_H
#define ESSENTIA_TCTOTOTAL_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class TCToTotal : public Algorithm {

 protected:
  Input<std::vector<Real> > _envelope;
  Output<Real> _TCToTotal;

 public:
  TCToTotal() {
    declareInput(_envelope, "envelope", "the envelope of the signal (its length must be greater than 1");
    declareOutput(_TCToTotal, "TCToTotal", "the temporal centroid to total length ratio");
  }

  ~TCToTotal() {}

  void configure() {}
  void compute();
  void declareParameters() {}

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#include "accumulatoralgorithm.h"

namespace essentia {
namespace streaming {

class TCToTotal : public AccumulatorAlgorithm {

 protected:
  Sink<Real> _envelope;
  Source<Real> _TCToTotal;

  int _idx;
  double _num, _den;

 public:
  TCToTotal() {
    declareInputStream(_envelope, "envelope", "the envelope of the signal (its length must be greater than 1");
    declareOutputResult(_TCToTotal, "TCToTotal", "the temporal centroid to total length ratio");
    reset();
  }

  void declareParameters() {}

  void reset();

  void consume();
  void finalProduce();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TCTOTOTAL_H
