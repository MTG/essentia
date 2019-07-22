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

#include "medianfilter.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char *MedianFilter::name = "MedianFilter";
const char *MedianFilter::category = "Filters";
const char *MedianFilter::description =
    DOC("This algorithm computes the median filtered version of the input "
        "signal giving the kernel size as detailed in [1].\n"
        "\n"
        "References:\n"
        "  [1] Median Filter -- from Wikipedia.org, \n"
        "  https://en.wikipedia.org/wiki/Median_filter");

void MedianFilter::configure() {
  _kernelSize = parameter("kernelSize").toInt();

  if (_kernelSize % 2 != 1)
    throw(EssentiaException("MedianFilter: kernelSize has to be odd"));
}

void MedianFilter::compute() {
  const std::vector<Real> &input = _array.get();
  std::vector<Real> &output = _filteredArray.get();

  int inputSize = input.size();
  int paddingSize = _kernelSize / 2;

  if (_kernelSize >= inputSize)
    throw(
        EssentiaException("kernelSize has to be smaller than the input size"));
  output.resize(inputSize);

  // add padding at the beginning and end so the ouput fits the input size.
  std::vector<Real> paddedArray = input;
  paddedArray.insert(paddedArray.begin(), paddingSize, input[0]);
  paddedArray.insert(paddedArray.end(), paddingSize, input.back());

  std::vector<Real>::const_iterator first = paddedArray.begin();
  std::vector<Real> window;
  for (int i = 0; i < inputSize; i++) {
    window.assign(first + i, first + i + _kernelSize);
    output[i] = median(window);
  }
}
