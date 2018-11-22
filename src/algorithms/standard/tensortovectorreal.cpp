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
const char* TensorToVectorReal::description = DOC("This algorithm streams the frames "
"of the input tensor along a given namespace ");


void TensorToVectorReal::configure() {
  _timeAxis = parameter("timeAxis").toInt();

  _batchSize = 0;
  _timeStamps = 0;
  _featsSize = 0;
} 


void TensorToVectorReal::reset() {
  _batchSize = 0;
  _timeStamps = 0;
  _featsSize = 0;
} 


AlgorithmStatus TensorToVectorReal::process() {
  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired (in: " << _frame.acquireSize()
            << " - out: " << _tensor.acquireSize() << ")");

  if (status != OK) {
      return status;
  };
  
  const_multi_array_ref<Real, 3> tensor = *(multi_array<Real, 3> *)_tensor.getFirstToken();

  if ((_batchSize != tensor.size()) || (_timeStamps != tensor.shape()[_timeAxis])) {
    EXEC_DEBUG("resizing frame acquire size");
    _batchSize = tensor.size();
    _timeStamps = tensor.shape()[_timeAxis];
    _featsSize = tensor.shape()[2];

    _frame.setAcquireSize(_timeStamps * _batchSize);
    _frame.setReleaseSize(_timeStamps * _batchSize);

    return process();
  }

  vector<vector<Real> >& frame = _frame.tokens();
  // TODO: This block could be cleaned up by taking adventage
  // of the boost view interfaces. Plus it allows to use and 
  // artbitrary time axis instead of a harcored one (without
  // using the [] operator)
  size_t i = 0;
  for (size_t j = 0; j < _batchSize; j++) {
    for (size_t k = 0; k < _timeStamps; k++, i++) {
      frame[i].resize(_featsSize);
      fastcopy(&frame[i][0], tensor[j][k].origin(), _featsSize);
    }
  }

  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");

  return OK;
}

} // namespace streaming
} // namespace essentia
