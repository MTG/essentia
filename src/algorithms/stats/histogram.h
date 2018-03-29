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

#ifndef ESSENTIA_HISTOGRAM_H
#define ESSENTIA_HISTOGRAM_H

#include "algorithmfactory.h"
#include <complex>

namespace essentia {
namespace standard {

class Histogram : public Algorithm {

 protected:
  Input<std::vector<Real> _inputArray;
  Output<std::vector<Real> > _histogramArray;

  string normMode;
  unsigned int numBins;
  double minRange;
  double maxRange;


 public:
  Histogram() {
    declareInput(_array, "inputArray", "the input array (cannot contain negative values, and must be non-empty)");
    declareOutput(_histogramArray, "histogramArray", "Array formatted for histogrsam plots (bins and their respective frequencies)");

  }


  void declareParameters() {
    declareParameter("numBins", "Number of required equal-width for the histogram", "[1,inf)", 10);
    declareParameter("normMode", "Type of normalization to be applied on the input array", "{None, unit_max, unit_sum}", "unit_max");
    declareParameter("minRange", "minimum range for the normalized array", "(0,inf)", 0);
    declareParameter("maxRange", "maximum range for the normalized array", "(0,inf)", 1);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Histogram : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _inputArray;
  Source<std::vector<Real> > _histogramArray;

 public:
  Histogram() {
    declareAlgorithm("Histogram");
    declareInput(_inputArray, TOKEN, "inputArray");
    declareOutput(_histogramArray, TOKEN, "histogramArray");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_HISTOGRAM_H
