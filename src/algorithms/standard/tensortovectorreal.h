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

#ifndef ESSENTIA_TENSORTOVECTORREAL_H
#define ESSENTIA_TENSORTOVECTORREAL_H

// #include "streamingalgorithmcomposite.h"
#include "streamingalgorithm.h"
#include "vectoroutput.h"

namespace essentia {
namespace streaming {

class TensorToVectorReal : public Algorithm {
 protected:
  Sink<Tensor<Real> > _tensor;
  Source<std::vector<Real> > _frame;

  int _timeAxis;
  int _batchSize;
  int _channels;
  int _timeStamps;
  int _featsSize;

 public:
  TensorToVectorReal(){
    declareInput(_tensor, 1, "tensor", "the input tensor");
    declareOutput(_frame, 128, "frame", "the frames to be retrieved from the tensor");
    _frame.setBufferType(BufferUsage::forMultipleFrames);
  }
  
  void declareParameters() {};

  void configure();
  void reset();
  AlgorithmStatus process();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TENSORTOVECTORREAL_H
