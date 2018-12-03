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
#ifndef ESSENTIA_COVERSONGSIMILARITY_H
#define ESSENTIA_COVERSONGSIMILARITY_H
#include "algorithmfactory.h"
#include <complex>

namespace essentia {
namespace standard {

 class CoverSongSimilarity : public Algorithm {
  protected:
   Input<std::vector<std::vector<Real> > > _inputArray;
   Output<std::vector<std::vector<Real> > > _scoreMatrix;
   Real disOnset;
   Real disExtension;
  public:
   CoverSongSimilarity() {
     declareInput(_inputArray, "inputArray", " a 2D binary cross similarity matrix of two audio chroma vectors (refer CrossSimilarityMatrix algorithm').");
     declareOutput(_scoreMatrix, "scoreMatrix", "2D cover song similarity score matrix from the input binary cross similarity matrix");
   }

   void declareParameters() {
     declareParameter("disOnset", "penalty for disruption onset", "[0,inf)", 0.5);
     declareParameter("disExtension", "penalty for disruption extension", "[0,inf)", 0.5);
     declareParameter("simType", "type of cover song similarity measure", "{qmax, dmax}", "qmax");
   }

   void configure();
   void compute();
   static const char* name;
   static const char* category;
   static const char* description;

  protected:
   Real _disOnset;
   Real _disExtension;
   enum SimType {
     QMAX, DMAX
   };
   SimType _simType;
   Real gammaState(Real value, const Real disOnset, const Real disExtension) const;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class CoverSongSimilarity : public StreamingAlgorithmWrapper {
  protected:
   Sink<std::vector<std::vector<Real> > > _inputArray;
   Source<std::vector<std::vector<Real> > > _scoreMatrix;
  public:
   CoverSongSimilarity() {
    declareAlgorithm("CoverSongSimilarity");
    declareInput(_inputArray, TOKEN, "inputArray");
    declareOutput(_scoreMatrix, TOKEN, "scoreMatrix");
}
};

} // namespace streaming
} // namespace essentia
#endif // ESSENTIA_COVERSONGSIMILARITY_H
