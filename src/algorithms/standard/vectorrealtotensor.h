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

  std::vector<int> _shape;
  int _timeStamps;
  int _batchHopSize;
  int _patchHopSize;
  bool _push;
  bool _accumulate;
  std::string _lastPatchMode;

  std::vector<std::vector<std::vector<Real> > > _acc;

 public:
  VectorRealToTensor(){
    declareInput(_frame, 187,"frame", "the input frames");
    declareOutput(_tensor, 1, "tensor", "the accumulated frame in one single tensor");
  }

  void declareParameters() {
    // Process 187 frames x 96 features by default.
    // This is a common setup for mel-spectrogram based arquitectures.
    std::vector<int> outputShape = {1, 1, 187, 96};

    declareParameter("shape", "shape of the output tensor (batchSize, channels, patchSize, featureSize). If batchSize is -1 a single tensor is generated when the end of the stream is reached", "", outputShape);
    declareParameter("patchHopSize", "number of frames between the beginnings of adjacent patches. 0 to avoid overlap", "[0,inf)", 0);
    declareParameter("batchHopSize", "number of patches between the beginnings of adjacent batches. 0 to avoid overlap", "[0,inf)", 0);
    declareParameter("lastPatchMode", "what to do with the last frames. Options are to `repeat` them to fill the last patch or to `discard` them", "{discard,repeat}", "repeat");
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
