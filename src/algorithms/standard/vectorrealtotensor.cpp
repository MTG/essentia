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

#include "vectorrealtotensor.h"

using namespace std;
using namespace boost;

namespace essentia {
namespace streaming {

const char* VectorRealToTensor::name = "VectorRealToTensor";
const char* VectorRealToTensor::category = "Standard";
const char* VectorRealToTensor::description = DOC("This algorithm takes a stream of frames "
"and outputs them as a single tensor everytime the output shape is reached. ")
"If the batch dimension (0) is set to -1 the data will be accumulated and returned "
"as a single tensor at the end of the stream.";


void VectorRealToTensor::configure() {
  vector<int> shape = parameter("shape").toVectorInt();
  _patchHopSize = parameter("patchHopSize").toInt();
  _batchHopSize = parameter("batchHopSize").toInt();
  _lastPatchMode = parameter("lastPatchMode").toString();

  for (size_t i = 0; i < shape.size(); i++) {
    _shape[i] = shape[i];
  }
  
  _timeStamps = shape[2];

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
  _pushedEverything = false;

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
  if ((int)_timeStamps != _frame.acquireSize()) {
    _frame.setAcquireSize(_timeStamps);
  }
  if ((int)_patchHopSize != _frame.releaseSize()) {
    _frame.setReleaseSize(_patchHopSize);
  }

  // Check if we have enough frames to add a patch.
  size_t available = _frame.available();
  bool addPatch = (_timeStamps <= available);

  // If we should stop just take the remaining frames.
  if (shouldStop() && (available < _timeStamps)) {
    _frame.setAcquireSize(available);
    _frame.setReleaseSize(available);
  
    // If we have been accumulating it is time to push!
    if (_accumulate) {
      // This is necesary to prevent and infinite loop.
      if (!_pushedEverything) {
        addPatch = true;
        _push = true;
      }
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

    if (frame.size() == _timeStamps) {
      _acc.push_back(frame);
    } else {
      // Rather repeat frames or discard the last patch.
      if (shouldStop()) {
        if (_lastPatchMode == "repeat") {
          size_t padAmount = _timeStamps - frame.size();

          vector<vector<Real> > frame_padded = frame;

          for (size_t i = 0; i < padAmount; i++) {
            frame_padded.push_back(frame[i % frame.size()]); 
          }

          _acc.push_back(frame_padded);
        }

        if (_lastPatchMode == "discard") {
          EXEC_DEBUG("discarding last frames");
        }

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
    boost::array<uint, 4> shape = _shape;
    size_t batchHopSize = _batchHopSize;

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
    
      _pushedEverything = true;
    }

    Tensor<Real>* tensor = (Tensor<Real> *)_tensor.getFirstToken();
    tensor->resize(shape);
    tensor->reshape(shape);

    // TODO: Add flag to swap frequency axis from 4 to 2.
    for (size_t i = 0; i < shape[0]; i++) {      // Batch axis
      for (size_t j = 0; j < shape[2]; j++) {    // Time axis
        fastcopy(&(*tensor)[i][0][j][0], &_acc[i][j][0], _acc[i][j].size());
      }
    }

    // Empty the accumulator.
    _acc.erase(_acc.begin(), _acc.begin() + batchHopSize);
  
    _push = false;
    outStatus = OK;
  }

  // Check if we should push in the next process().
  if (!_accumulate) {
    if (_acc.size() >= _shape[0]) _push = true;
  }
  
  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");
  return outStatus;
}

void VectorRealToTensor::reset() {
  _acc.assign(0, vector<vector<Real> >(_shape[1], vector<Real>(_shape[2], 0.0)));
  _push = false;
  _pushedEverything = false;
}

} // namespace streaming
} // namespace essentia
