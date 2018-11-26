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

#ifndef ESSENTIA_VECTORREALTOTENSOR_H
#define ESSENTIA_VECTORREALTOTENSOR_H

#include "streamingalgorithm.h"
#include "vectoroutput.h"

namespace essentia {
namespace streaming {

class VectorRealToTensor : public Algorithm {
 protected:
  Sink<std::vector<Real> > _frame;
  Source<Tensor<Real> > _tensor;

  boost::array<uint, 4> _shape;
  int _timeAxis;
  size_t _timeStamps;
  size_t _batchHopSize;
  size_t _patchHopSize;
  bool _push;
  bool _accumulate;
  bool _pushedEverything;
  std::string _lastPatchMode;

  std::vector<std::vector<std::vector<Real> > > _acc;

 public:
  VectorRealToTensor(){
    declareInput(_frame, 128,"frame", "the input frames");
    declareOutput(_tensor, 1, "tensor", "the accumulated frame in one single tensor");

  }

  void declareParameters() {
    // TODO: set a better default shape
    std::vector<int> outputShape = {-1, 1, 128, 128};
    declareParameter("shape", "the size of output tensor. If batch dimension (the first one) is -1 it will accumulate as many batches as available in the input track", "", outputShape);
    declareParameter("timeAxis", "the axis where the frames will be stacked", "(0,inf)", 2);
    declareParameter("patchHopSize", "number of frames between the beginnings of adjacent patches. 0 to avoid overlap", "[0,inf)", 0);
    declareParameter("batchHopSize", "number of patches between the beginnings of adjacent batches. 0 to avoid overlap", "[0,inf)", 0);
    declareParameter("lastPatchMode", "what to do with the last incomplete patch", "{discard,repeat}", "repeat");
  }

  void configure();
  void reset();
  AlgorithmStatus process();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_VECTORREALTOTENSOR_H