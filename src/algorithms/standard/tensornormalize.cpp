/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#include "tensornormalize.h"
#include <sstream>

using namespace essentia;
using namespace standard;

using namespace std;

const char* TensorNormalize::name = "TensorNormalize";
const char* TensorNormalize::category = "Standard";
const char* TensorNormalize::description = DOC("This algorithm performs normalization over a tensor.\n"
"When the axis parameter is set to -1 the input tensor is globally normalized. Any other value means"
" that the tensor will be normalized along that axis.\n"
"This algorithm supports Standard and MinMax normalizations.\n"
"\n"
"References:\n"
"  [1] Feature scaling - Wikipedia, the free encyclopedia,\n"
"  https://en.wikipedia.org/wiki/Feature_scaling");


  TensorNormalize::Scaler TensorNormalize::scalerFromString(const std::string& name) const {
    if (name == "standard") return STANDARD;
    if (name == "minMax") return MINMAX;

    throw EssentiaException("TensorNormalize: Unknown scaler type: ", name);
  }

  void TensorNormalize::configure() {
    _scaler = scalerFromString(parameter("scaler").toString());
    _axis = parameter("axis").toInt();
  }

void TensorNormalize::compute() {
  const Tensor<Real>& input = _input.get();
  Tensor<Real>& output = _output.get();

  switch (_scaler) {
    case STANDARD: {
      if (_axis == -1) {
        Real globalMean = mean(input);
        Real globalStd = stddev(input, globalMean);

        if (globalStd == 0) {
          E_INFO("TensorNormalize: Received tensor with constant value.");
          // Set std to 1 so we return a vector of all 0s.
          globalStd = 1;
        }

        output = (input - globalMean) / globalStd;

      } else {
        Tensor<Real> means = mean(input, _axis);
        Tensor<Real> stds = stddev(input, means, _axis);

        bool hasConstantSlice = false;

        Real* stds_data = stds.data();
        for (int i = 0; i < stds.size(); i++) {
          if (stds_data[i] == 0) {
            stds_data[i] = 1;
            hasConstantSlice = true;
          }
        }

        if (hasConstantSlice) {
          E_INFO("TensorNormalize: Received tensor with constant value.");
        }

        array<Eigen::Index, TENSORRANK> broadcastShape = input.dimensions();
        broadcastShape[_axis] = 1;

        output = (input - means.broadcast(broadcastShape)) / stds.broadcast(broadcastShape);
      }
      break;
    }
    case MINMAX: {
      if (_axis == -1) {
        Real globalMinimum = tensorMin(input);
        Real globalMaximum = tensorMax(input);

        output = input - globalMinimum;

        if (globalMaximum == globalMinimum) {
          E_INFO("TensorNormalize: Received tensor with constant value.");
        } else {
          output /= output.constant(globalMaximum - globalMinimum);
        }

      } else {
        Tensor<Real> minima = tensorMin(input, _axis);
        Tensor<Real> maxima = tensorMax(input, _axis);

        array<Eigen::Index, TENSORRANK> broadcastShape = input.dimensions();
        broadcastShape[_axis] = 1;

        Tensor<Real> diff = maxima - minima;

        bool hasConstantSlice = false;

        Real* diff_data = diff.data();
        for (int i = 0; i < diff.size(); i++) {
          if (diff_data[i] == 0) {
            diff_data[i] = 1;
            hasConstantSlice = true;
          }
        }

        if (hasConstantSlice) {
          E_INFO("TensorNormalize: Received tensor with constant value.");
        }

        output = (input - minima.broadcast(broadcastShape)) / diff.broadcast(broadcastShape);
      }
      break;
    }
    default: {
      throw EssentiaException("TensorNormalize: Unknown scaler type.");
    }
  }

}
