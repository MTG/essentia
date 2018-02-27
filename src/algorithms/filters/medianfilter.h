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

#ifndef ESSENTIA_MEDIANFILTER_H
#define ESSENTIA_MEDIANFILTER_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class MedianFilter : public Algorithm {
  
 private:
  Input<std::vector<Real>> _array;
  Output<std::vector<Real>> _filteredArray;

  int _kernelSize;

 public:
  MedianFilter() {
    declareInput(_array, "array", "the input array (must be non-empty)");
    declareOutput(_filteredArray, "filteredArray",
                  "the median-filtered input array");
  }

  void declareParameters() {
    declareParameter("kernelSize", "scalar giving the size of the median filter window. Must be odd", "[1,inf)", 11);
  }

  void configure();
  void compute();

  static const char *name;
  static const char *category;
  static const char *description;
};

}  // namespace standard
}  // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class MedianFilter : public StreamingAlgorithmWrapper {
 protected:
  Sink<std::vector<Real>> _array;
  Source<std::vector<Real>> _filteredArray;

 public:
  MedianFilter() {
    declareAlgorithm("MedianFilter");
    declareInput(_array, TOKEN, "array");
    declareOutput(_filteredArray, TOKEN, "filteredArray");
  }
};

}  // namespace streaming
}  // namespace essentia

#endif // ESSENTIA_MEDIANFILTER_H
