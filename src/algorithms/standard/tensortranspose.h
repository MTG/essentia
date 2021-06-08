/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_TENSORTRANSPOSE_H
#define ESSENTIA_TENSORTRANSPOSE_H

#include "algorithm.h"

namespace essentia {
namespace standard {


class TensorTranspose : public Algorithm {

 protected:
  Input<Tensor<Real> > _input;
  Output<Tensor<Real> > _output;

  std::vector<int> _permutation;

 public:
  TensorTranspose() {
    declareInput(_input, "tensor", "the input tensor");
    declareOutput(_output, "tensor", "the transposed output tensor");
  }

  void declareParameters() {
    declareParameter("permutation", "permutation of [0,1,2,3]. The i'th dimension of the returned tensor will correspond to the dimension numbered permutation[i] of the input.", "", Parameter::VECTOR_INT);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class TensorTranspose : public StreamingAlgorithmWrapper {

 protected:
  Sink<Tensor<Real> > _input;
  Source<Tensor<Real> > _output;

 public:
  TensorTranspose() {
    declareAlgorithm("TensorTranspose");
    declareInput(_input, TOKEN, "tensor");
    declareOutput(_output, TOKEN, "tensor");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TENSORTRANSPOSE_H
