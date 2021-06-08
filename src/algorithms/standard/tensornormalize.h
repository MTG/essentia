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

#ifndef ESSENTIA_TENSORNORMALIZE_H
#define ESSENTIA_TENSORNORMALIZE_H

#include "algorithm.h"
#include "essentiamath.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace essentia {
namespace standard {


class TensorNormalize : public Algorithm {

 protected:
  enum Scaler {
    STANDARD,
    MINMAX
  };

  Scaler scalerFromString(const std::string& name) const;

  Input<Tensor<Real> > _input;
  Output<Tensor<Real> > _output;

  Scaler _scaler;
  int _axis;

 public:
  TensorNormalize() {
    declareInput(_input, "tensor", "the input tensor");
    declareOutput(_output, "tensor", "the normalized output tensor");
  }

  void declareParameters() {
    declareParameter("scaler", "the type of the normalization to apply to input tensor", "{standard,minMax}", "standard");
    declareParameter("axis", "Normalize along the given axis. -1 to normalize along all the dimensions", "[-1, 4)", 0);
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

class TensorNormalize : public StreamingAlgorithmWrapper {

 protected:
  Sink<Tensor<Real> > _input;
  Source<Tensor<Real> > _output;

 public:
  TensorNormalize() {
    declareAlgorithm("TensorNormalize");
    declareInput(_input, TOKEN, "tensor");
    declareOutput(_output, TOKEN, "tensor");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TENSORNORMALIZE_H
