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
#ifndef ESSENTIA_CROSSSIMILARITYMATRIX_H
#define ESSENTIA_CROSSSIMILARITYMATRIX_H
#include "algorithmfactory.h"
#include <complex>

namespace essentia {
namespace standard {

class CrossSimilarityMatrix : public Algorithm {
  protected:
   Input<std::vector<std::vector<Real> > > _queryFeature;
   Input<std::vector<std::vector<Real> > > _referenceFeature;
   Output<std::vector<std::vector<Real> > > _csm;
  public:
   CrossSimilarityMatrix() {
    declareInput(_queryFeature, "queryFeature", "input frame features of the query song (e.g., a chromagram)");
    declareInput(_referenceFeature, "referenceFeature", "input frame features of the reference song (e.g., a chromagram)");
    declareOutput(_csm, "csm", "2D cross-similarity matrix of two input frame sequences (query vs reference)");
   }

   void declareParameters() {
    declareParameter("frameStackStride", "stride size to form a stack of frames (e.g., 'frameStackStride'=1 to use consecutive frames; 'frameStackStride'=2 for using every second frame)", "[1,inf)", 1);
    declareParameter("frameStackSize", "number of input frames to stack together and treat as a feature vector for similarity computation. Choose 'frameStackSize=1' to use the original input frames without stacking", "[0,inf)", 1);
    declareParameter("binarizePercentile", "maximum percent of distance values to consider as similar in each row and each column", "[0,1]", 0.095);
    declareParameter("binarize", "whether to binarize the euclidean cross-similarity matrix", "{true,false}", false);
  }

   void configure();
   void compute();

   static const char* name;
   static const char* category;
   static const char* description;

  protected:
   int _frameStackStride;
   int _frameStackSize;
   Real _binarizePercentile;
   bool _binarize;
   bool _status;
   std::vector<std::vector<Real> > stackFrames(std::vector<std::vector<Real> >& frames, int frameStackSize, int frameStackStride) const;
   std::vector<Real> getColsAtVecIndex(std::vector<std::vector<Real> >& inputMatrix, int index) const;
};

} // namespace standard
} // namespace essentia
 #endif // ESSENTIA_CROSSSIMILARITYMATRIX_H
