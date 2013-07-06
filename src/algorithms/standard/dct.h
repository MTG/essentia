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

#ifndef ESSENTIA_DCT_H
#define ESSENTIA_DCT_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class DCT : public Algorithm {

 protected:
  Input<std::vector<Real> > _array;
  Output<std::vector<Real> > _dct;

 public:
  DCT() {
    declareInput(_array, "array", "the input array");
    declareOutput(_dct, "dct", "the discrete cosine transform of the input array");
  }

  void declareParameters() {
    declareParameter("inputSize", "the size of the input array", "[1,inf)", 10);
    declareParameter("outputSize", "the number of output coefficients", "[1,inf)", 10);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;


 protected:
  int _outputSize;
  void createDctTable(int inputSize, int outputSize);

  std::vector<std::vector<Real> > _dctTable;
};

} // namespace essentia
} // namespace standard


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class DCT : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<std::vector<Real> > _dct;

 public:
  DCT() {
    declareAlgorithm("DCT");
    declareInput(_array, TOKEN, "array");
    declareOutput(_dct, TOKEN, "dct");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_DCT_H
