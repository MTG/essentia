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
   Output<Real> _distance;
   Real disOnset;
   Real disExtension;
  public:
   CoverSongSimilarity() {
     declareInput(_inputArray, "inputArray", " a 2D binary cross-similarity matrix of two audio chroma vectors (query vs reference song) (refer 'ChromaCrossSimilarity' algorithm').");
     declareOutput(_scoreMatrix, "scoreMatrix", "a 2D smith-waterman alignment score matrix from the input binary cross-similarity matrix");
     declareOutput(_distance, "distance", "asymmetric cover song similarity distance between the query and reference song from the input similarity matrix as described in [2].");
   }

   void declareParameters() {
     declareParameter("disOnset", "penalty for disruption onset", "[0,inf)", 0.5);
     declareParameter("disExtension", "penalty for disruption extension", "[0,inf)", 0.5);
     declareParameter("alignmentType", "choose either one of the given local-alignment constraints for smith-waterman algorithm as described in [2] and [3] respectively.", "{serra09, chen17}", "serra09");
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
     SERRA09, CHEN17
   };
   SimType _simType;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class CoverSongSimilarity : public Algorithm {
  protected:
   Sink<std::vector<Real> > _inputArray;
   Source<TNT::Array2D<Real> > _scoreMatrix;
   Source<Real> _distance;
   
   // params and global protected variables
   int _minFramesSize = 2;
   int _iterIdx = 0;
   Real _disOnset;
   Real _disExtension;
   Real _c1;
   Real _c2;
   Real _c3;
   Real _c4;
   Real _c5;
   size_t _xFrames;
   size_t _yFrames;
   size_t _xIter;
   size_t _accumXFrameSize;
   size_t _x;
   std::vector<std::vector<Real> > _prevCumMatrixFrames;
   std::vector<std::vector<Real> > _previnputMatrixFrames;
   std::vector<std::vector<Real> > _bufferScoreMatrix;

  public:
   CoverSongSimilarity() : Algorithm() {
    declareInput(_inputArray, _minFramesSize, "inputArray", "a 2D binary cross similarity matrix of two audio chroma vectors (refer CrossSimilarityMatrix algorithm').");
    declareOutput(_scoreMatrix, 1, "scoreMatrix", "a 2D smith-waterman alignment score matrix from the input binary cross-similarity matrix as described in [2].");
    declareOutput(_distance, 1, "distance", "asymmetric cover song similarity distance between the query and reference song from the input similarity matrix as described in [2].");
  }

  ~CoverSongSimilarity() {}

  AlgorithmStatus process();

  void declareParameters() {
    declareParameter("disOnset", "penalty for disruption onset", "[0,inf)", 0.5);
    declareParameter("disExtension", "penalty for disruption extension", "[0,inf)", 0.5);
  }

  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace streaming
} // namespace essentia
#endif // ESSENTIA_COVERSONGSIMILARITY_H

