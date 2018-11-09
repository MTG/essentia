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

#include "tensortovectorreal.h"

using namespace std;
using namespace boost;

namespace essentia {
namespace streaming {

const char* TensorToVectorReal::name = "TensorToVectorReal";
const char* TensorToVectorReal::category = "Standard";
const char* TensorToVectorReal::description = DOC("This algorithm takes a stream of tensors "
"and outputs them as a stream of frames");

void TensorToVectorReal::configure() {
  vector<int> shape = parameter("shape").toVectorInt();
  _timeAxis = parameter("timeAxis").toInt();

  _timeStamps = shape[_timeAxis];
  _frame.setAcquireSize(_timeStamps);
  _frame.setReleaseSize(_timeStamps);
} 


AlgorithmStatus TensorToVectorReal::process() {
  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired (in: " << _frame.acquireSize()
            << " - out: " << _tensor.acquireSize() << ")");

  if (status != OK) {
      return status;
  };
  
  const vector<multi_array<Real, 3> >& tensor = _tensor.tokens();

  vector<vector<Real> >& frame = _frame.tokens();

  // TODO: This block could be cleaned up by taking adventage
  // of the boost view interfaces. Plus it allows to use and 
  // artbitrary time axis instead of a harcored one (without
  // using the [] operator)  
  size_t frameIdx;  
  for (size_t i = 0; i < tensor.size(); i++) {
    for (size_t j = 0; j < _timeStamps; j++) {
      frameIdx = i * _timeStamps + j;
      frame[frameIdx].resize((int)tensor[i][0][j].size());
      fastcopy(&frame[frameIdx][0],
               tensor[i][0][j].origin(), 
               (int)tensor[i][0][j].size());
    }
  }

  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");

  return OK;
}

} // namespace streaming
} // namespace essentia
