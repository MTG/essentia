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

#include "algorithm.h" 

namespace essentia {
namespace standard {

class Histogram : public Algorithm {

 protected:
  Input<std::vector<Real> > _array;
  Output<std::vector<Real> > _histogram;
  Output<std::vector<Real> > _binEdges;

  std::string  _normalize;
  Real _minValue;
  Real _maxValue;
  int  _numberBins;

 private:
  Real binWidth;
  std::vector<Real> tempBinEdges;  

 public:
  Histogram() {
    declareInput(_array, "array", "the input array");
    declareOutput(_histogram, "histogram", "the values in the equally-spaced bins");
    declareOutput(_binEdges, "binEdges", "the edges of the equally-spaced bins. Size is _histogram.size() + 1");
  }

  void declareParameters(){
    declareParameter("normalize", "the normalization setting.", "{none,unit_sum,unit_max}",  std::string("none"));
    declareParameter("minValue", "the min value of the histogram", "[0, Inf)", 0.0);
    declareParameter("maxValue", "the max value of the histogram", "[0, Inf)", 1.0);
    declareParameter("numberBins", "the number of bins", "(0, Inf)", 10);
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
  Sink<std::vector<Real> > _array;
  Source<std::vector<Real> > _histogram;
  Source<std::vector<Real> > _binEdges;

 public:
  Histogram() {
   declareAlgorithm("Histogram");
   declareInput(_array, TOKEN, "array");
   declareOutput(_histogram, TOKEN, "histogram");
   declareOutput(_binEdges, TOKEN, "binEdges");
  }
};

} //namespace streaming
} //namespace essentia

#endif //ESSENTIA_HISTOGRAM_H



