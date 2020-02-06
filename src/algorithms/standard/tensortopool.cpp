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

#include "tensortopool.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* TensorToPool::name = "TensorToPool";
const char* TensorToPool::category = "Standard";
const char* TensorToPool::description = DOC("This algorithm inserts a tensor into "
"a pool under a given namespace. Suppors adding (accumulation) or (setting) overwritting");

void TensorToPool::configure() {
  _mode = parameter("mode").toString();
  _namespace = parameter("namespace").toString();
}

AlgorithmStatus TensorToPool::process() {
  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired (in: " << _tensor.acquireSize()
             << " - out: " << _pool.acquireSize() << ")");

  if (status != OK) {
    return status;
  };

  const vector<Tensor<Real> >& tensor = _tensor.tokens();
  vector<Pool>& pool = _pool.tokens();

  if (_mode == "add") {
    for (size_t i = 0; i < tensor.size(); i++) {
      pool[i].add(_namespace, tensor[i]);
    }
  }

  else if (_mode == "overwrite") {
    for (size_t i = 0; i < tensor.size(); i++) {
      pool[i].set(_namespace, tensor[i]);
    }
  }

  else {
    throw EssentiaException("TensorToPool: Invalid operation mode.");
  }

  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");

  return OK;
}

} // namespace streaming
} // namespace essentia
