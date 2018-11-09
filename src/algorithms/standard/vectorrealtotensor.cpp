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
"and outputs them as a single tensor everytime the output shape is reached.");

void VectorRealToTensor::configure() {
  vector<int> shape = parameter("shape").toVectorInt();
  _timeAxis = parameter("timeAxis").toInt();

  _timeStamps = shape[_timeAxis];

  EXEC_DEBUG("Resizing input buffers");  
  _frame.setAcquireSize(_timeStamps);
  _frame.setReleaseSize(_timeStamps);

  for (size_t i = 0; i < shape.size(); i++) {
    _shape[i] = shape[i];
  }
}

AlgorithmStatus VectorRealToTensor::process() {
  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired (in: " << _frame.acquireSize()
            << " - out: " << _tensor.acquireSize() << ")");

  if (status != OK) {
    if (status == NO_INPUT){
      EXEC_DEBUG("still not eneugh frames to fill the tensor");
      return status;
    }  
      // handle other casees
      return FINISHED;
  };

  const vector<vector<Real> >& frame = _frame.tokens();
  vector<multi_array<Real, 3> >& tensor = _tensor.tokens();

  tensor[0].resize(_shape);
  tensor[0].reshape(_shape);

  // TODO: external loop to consider batch size also
  for (size_t i = 0; i < _timeStamps; i++) {
    
    // TODO: clean this taking adventage of  subarray references 
    for (size_t j = 0; j < frame[0].size(); j++) {
      tensor[0][0][i][j] = frame[i][j];
    }
  }

  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");

  return OK;
}

} // namespace streaming
} // namespace essentia
