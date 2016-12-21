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

#ifndef ESSENTIA_IDCT_H
#define ESSENTIA_IDCT_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class IDCT : public Algorithm {

 protected:
  Input<std::vector<Real> > _dct;
  Output<std::vector<Real> > _idct;

 public:
  IDCT() {
    declareInput(_dct, "dct", "the discrete cosine transform");
    declareOutput(_idct, "idct", "the inverse cosine transform of the input array");
  }

  void declareParameters() {
    declareParameter("inputSize", "the size of the input array", "[1,inf)", 10);
    declareParameter("outputSize", "the number of output coefficients", "[1,inf)", 10);
    declareParameter("dctType", "the DCT type", "[2,3]", 2);
    declareParameter("liftering", "the liftering coefficient. Use '0' to bypass it", "[0,inf)", 0);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;


 protected:
  int _outputSize;
  Real _lifter;
  void createIDctTableII(int inputSize, int outputSize);
  void createIDctTableIII(int inputSize, int outputSize);
  std::vector<std::vector<Real> > _idctTable;
  int _type;
};

} // namespace essentia
} // namespace standard


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class IDCT : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _dct;
  Source<std::vector<Real> > _idct;

 public:
  IDCT() {
    declareAlgorithm("IDCT");
    declareInput(_dct, TOKEN, "dct");
    declareOutput(_idct, TOKEN, "idct");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_IDCT_H
