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

#include "tensortranspose.h"
#include <sstream>

using namespace essentia;
using namespace standard;

using namespace std;

const char* TensorTranspose::name = "TensorTranspose";
const char* TensorTranspose::category = "Standard";
const char* TensorTranspose::description = DOC("This algorithm performs transpositions over the axes of a tensor.\n");

  void TensorTranspose::configure() {
    if (!parameter("permutation").isConfigured()) return;
    _permutation = parameter("permutation").toVectorInt();

    if (_permutation.size() != TENSORRANK) {
      throw EssentiaException("TensorTranspose: the size of the permutation vector is ",
                              _permutation.size(), " while it should be ", TENSORRANK );
    }

    int minimun = *min_element(_permutation.begin(), _permutation.end());
    if (minimun < 0) throw EssentiaException("TensorTranspose: one of the elements of the permutation vector was set to ",
                                             minimun, ", while the minimum value has to be be 0");

    int maximum = *max_element(_permutation.begin(), _permutation.end());
    if (maximum > TENSORRANK -1) throw EssentiaException("TensorTranspose: one of the elements of the permutation vector was set to ",
                                                          maximum, ", while the maximum value has to be ", TENSORRANK -1);

    for (int i = 0; i < TENSORRANK; i++) {
      if (!count(_permutation.begin(), _permutation.end(), i)) {
        throw EssentiaException("TensorTranspose: Index (", i, ") not found in `permutaiton`.");
      }
    }
    
  }

void TensorTranspose::compute() {
  const Tensor<Real>& input = _input.get();
  Tensor<Real>& output = _output.get();

  output = input.shuffle(_permutation);
}
