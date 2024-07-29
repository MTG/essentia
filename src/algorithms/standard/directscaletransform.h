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

#ifndef ESSENTIA_DIRECTSCALETRANSFORM_H
#define ESSENTIA_DIRECTSCALETRANSFORM_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class DirectScaleTransform : public Algorithm {

 protected:
   Input<std::vector<std::vector<Real>> > _matrix;
   Output<std::vector<std::vector<Real>> > _result;

   Real _C;
   Real _fs;

 public:
    DirectScaleTransform() {
      declareInput(_matrix, "matrix", "the input matrix");
      declareOutput(_result, "result", "the result of the direct scale transform");
    }
    
    void declareParameters() {
      declareParameter("C", "desired scale for the direct scale transform", "(0,inf)", 500);
      declareParameter("fs", "the sampling rate of the input autocorrelation", "(0,inf)", 1);
    }
    
    void configure();
    void compute();

  static const char* name;
  static const char* category;
  static const char* description;

};

}  // namespace standard
}  // namespace essentia

#include <streamingalgorithmwrapper.h>

namespace essentia {
namespace streaming {

class DirectScaleTransform : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::vector<Real>>> _matrix;
  Source<std::vector<std::vector<Real>>> _result;

 public:
    DirectScaleTransform() {
        declareAlgorithm("DirectScaleTransform");
        declareInput(_matrix, TOKEN, "the input matrix");
        declareOutput(_result, TOKEN, "the result of the direct scale transform");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_DIRECTSCALETRANSFORM_H
