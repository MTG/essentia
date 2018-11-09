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

#include "pooltotensor.h"

using namespace std;
using namespace boost;

namespace essentia {
namespace streaming {

const char* PoolToTensor::name = "PoolToTensor";
const char* PoolToTensor::category = "Standard";
const char* PoolToTensor::description = DOC("This algorithm retrieve tensors from "
"pools under a given namespace");

void PoolToTensor::configure() {
  _namespace = parameter("namespace").toString();
}

AlgorithmStatus PoolToTensor::process() {
  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired (in: " << _tensor.acquireSize()
            << " - out: " << _pool.acquireSize() << ")");

  if (status != OK) {
      return status;
  };

  const vector<Pool>& pool = _pool.tokens();
  vector<multi_array<Real, 3> >& tensor = _tensor.tokens();

  for (size_t i = 0; i < tensor.size(); i++) {
    const_multi_array_ref<Real, 3> data(
      pool[i].value<vector<multi_array<Real, 3> > >(_namespace)[0]);

    // TODO: is there a more beatiful way to get the multi_array shape?
    // Otherwise a method getArrayShape() could be added to an 
    // Essentia child class.  
    auto& shape = reinterpret_cast<boost::array<size_t, const_multi_array_ref<Real, 3>::dimensionality> const&>(*data.shape());
    
    tensor[i].resize(shape);
    tensor[i].reshape(shape);
    tensor[i] = data;
  }

  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");

  return OK;
}

} // namespace streaming
} // namespace essentia
