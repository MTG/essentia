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

#include "vectorrealtotensor.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* VectorRealToTensor::name = "VectorRealToTensor";
const char* VectorRealToTensor::category = "Standard";
const char* VectorRealToTensor::description = DOC("This algorithm generates tensors "
"out of a stream of input frames. The 4 dimensions of the tensors stand for (batchSize, channels, patchSize, featureSize):\n"
"  - batchSize: Number of patches per tensor. If batchSize is set to 0 it will accumulate patches until the end of the stream is reached and then produce a single tensor. "
"Warning: This option may exhaust memory depending on the size of the stream.\n"
"  - channels: Number of channels per tensor. Currently, only single-channel tensors are supported. Otherwise, an exception is thrown.\n"
"  - patchSize: Number of timestamps (i.e., number of frames) per patch.\n"
"  - featureSize: Expected number of features (e.g., mel bands) of every input frame. This algorithm throws an exception if the size of any frame is different from featureSize.\n"
"Additionally, the patchHopSize and batchHopSize parameters provide control over the amount of overlap on those dimensions.");


void VectorRealToTensor::configure() {
  vector<int> shape = parameter("shape").toVectorInt();
  _patchHopSize = parameter("patchHopSize").toInt();
  _batchHopSize = parameter("batchHopSize").toInt();
  _lastPatchMode = parameter("lastPatchMode").toString();

  _shape.resize(shape.size());
  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] == 0) {
    throw EssentiaException("VectorRealToTensor: All dimensions should have a non-zero size.");
    }

    _shape[i] = shape[i];
  }

  if (shape[1] != 1) {
    throw EssentiaException("VectorRealToTensor: Currently only single-channel tensors are supported.");
  }

  _timeStamps = shape[2];
  _frame.setAcquireSize(_timeStamps);

  if (shape[0] == -1) {
    _accumulate = true;
  }

  if (_batchHopSize == 0) {
    _batchHopSize = shape[0];
  }

  if (_patchHopSize == 0) {
    _patchHopSize = _timeStamps;
  }

  _acc.assign(0, vector<vector<Real> >(_shape[2], vector<Real>(_shape[3], 0.0)));
  _push = false;

  if (_patchHopSize > _timeStamps) {
    throw EssentiaException("VectorRealToTensor: `patchHopSize` has to be smaller that the number of timestamps");
  }


  if (shape[0] > 0) {
    if (_batchHopSize > _timeStamps) {
      throw EssentiaException("VectorRealToTensor: `batchHopSize` has to be smaller that the number batch size (shape[0])");
    }
  }

}


AlgorithmStatus VectorRealToTensor::process() {
  EXEC_DEBUG("process()");
  if (_timeStamps != _frame.acquireSize()) {
    _frame.setAcquireSize(_timeStamps);
  }
  if (_patchHopSize != _frame.releaseSize()) {
    _frame.setReleaseSize(_patchHopSize);
  }

  // Check if we have enough frames to add a patch.
  int available = _frame.available();
  bool addPatch = (available >= _timeStamps);

  // If we should stop just take the remaining frames.
  if (shouldStop() && (available < _timeStamps)) {
    _frame.setAcquireSize(available);
    _frame.setReleaseSize(available);

    // Push if there are remaining frames
    if (_lastPatchMode == "repeat" && available > 0) {
      addPatch = true;
      _push = true;
  
    // or if we have been accumulating.
    } else if (_accumulate && _acc.size() >= 1) {
      addPatch = true;
      _push = true;
    }
  }

  // Return if there is nothing to do.
  if ((!addPatch) && (!_push)) return NO_INPUT;

  if (_push) {
    _tensor.setAcquireSize(1);
    _tensor.setReleaseSize(1);

    // Don't get frames if we just want to push.
    if (!addPatch) {
      _frame.setAcquireSize(0);
      _frame.setReleaseSize(0);
    }

  // Don't get a tensor if we are just accumulating.
  } else {
    _tensor.setAcquireSize(0);
    _tensor.setReleaseSize(0);
  }

  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired (in: " << _frame.acquireSize()
            << " - out: " << _tensor.acquireSize() << ")");

  if (status != OK) {
    return status;
  };

  AlgorithmStatus outStatus = NO_OUTPUT;

  // Frames accumulation step.
  if (addPatch) {
    const vector<vector<Real> >& frame = _frame.tokens();

    // Sanity check.
    for (size_t i = 0; i < frame.size(); i++) {
      if ((int)frame[i].size() != _shape[3]) {
        throw EssentiaException("VectorRealToTensor: Found input frame with size ", frame[i].size(),
                                " while the algorithm was configured to work with frames with size ", _shape[3]);
      }
    }

    // Add a regular patch.
    if ((int)frame.size() == _timeStamps) {
      _acc.push_back(frame);

    // If size does not match rather repeat frames or discard them.
    } else {
      if (_lastPatchMode == "repeat") {
        if (frame.size() == 0) {
          EXEC_DEBUG("VectorRealToTensor: 0 frames remaining.");

        } else {
          if (frame.size() < 10) {
            E_WARNING("VectorRealToTensor: Last patch produced by repeating the last " << frame.size() << " frames. May result in unreliable predictions.");
          }
          vector<vector<Real> > padded_frame = frame;

          for (int i = 0; i < _timeStamps; i++) {
            padded_frame.push_back(frame[i % frame.size()]);
          }

          EXEC_DEBUG("VectorRealToTensor: Repeating the remaining " << frame.size() << " frames to make one last patch.");
          _acc.push_back(padded_frame);
        }

      } else if (_lastPatchMode == "discard") {
        EXEC_DEBUG("VectorRealToTensor: Discarding last frames");

      } else {
        throw EssentiaException("VectorRealToTensor: Incomplete patch found "
                                "before reaching the end of the stream. This is not supposed to happen");
      }
    }
  }

  // We only push if when we have filled the whole batch
  // or if we have reached the end of the stream in
  // accumulate mode.
  if (_push) {
    vector<int> shape = _shape;
    int batchHopSize = _batchHopSize;

    // If we have been accumulating we have to get the
    // tensor's shape from the current status of the
    // accumulator.
    if (_accumulate) {
      shape[0] = _acc.size();
      batchHopSize = _acc.size();

      if (_acc.size() == 0) {
        throw EssentiaException("VectorRealToTensor: The stream has finished without enough frames to "
                                "produce a patch of the desired size. Consider setting the `lastPatchMode` "
                                "parameter to `repeat` in order to produce a batch.");
      }
    }

    Tensor<Real>& tensor = *(Tensor<Real> *)_tensor.getFirstToken();
    tensor.resize(shape);

    // TODO: Add flag to swap frequency axis from 4 to 2.
    for (int i = 0; i < shape[0]; i++) {      // Batch axis
      for (int j = 0; j < shape[2]; j++) {    // Time axis
        for (int k = 0; k < shape[3]; k++) {  // Freq axis
          tensor(i, 0, j, k) = _acc[i][j][k];
        }
      }
    }

    // Empty the accumulator.
    _acc.erase(_acc.begin(), _acc.begin() + batchHopSize);
  
    _push = false;
    outStatus = OK;
  }

  // Check if we should push in the next process().
  if (!_accumulate) {
    if ((int)_acc.size() >= _shape[0]) _push = true;
  }

  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");
  return outStatus;
}

void VectorRealToTensor::reset() {
  _acc.assign(0, vector<vector<Real> >(_shape[1], vector<Real>(_shape[2], 0.0)));
  _push = false;
}

} // namespace streaming
} // namespace essentia
