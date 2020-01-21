/*
 * Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_POOLTOTENSOR_H
#define ESSENTIA_POOLTOTENSOR_H

#include "streamingalgorithm.h"
#include "pool.h"

namespace essentia {
namespace streaming {

class PoolToTensor : public Algorithm {
 protected:
  Sink<Pool> _pool;
  Source<Tensor<Real> > _tensor;

  std::string _namespace;

 public:
  PoolToTensor(){
    declareInput(_pool, 1, "pool", "the pool containing the required namespace");
    declareOutput(_tensor, 1, "tensor", "tensors retrieved from the pool");
  }

  void declareParameters() {
    declareParameter("namespace", "the input namespace", "", "input_0");
  }

  void configure();
  AlgorithmStatus process();


  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_POOLTOTENSOR_H
