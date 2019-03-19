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
#include "crosssimilaritymatrix.h"
#include "essentia/utils/tnt/tnt2vector.h"
#include "essentiamath.h"
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <functional>

namespace essentia {
namespace standard {

const char* CrossSimilarityMatrix::name = "CrossSimilarityMatrix";
const char* CrossSimilarityMatrix::category = "Music Similarity";
const char* CrossSimilarityMatrix::description = DOC("This algorithm computes an euclidean cross-similarity matrix from two 2D input feature vectors.\n\n"
"In addition, the algorithm also provides an option to binarize the euclidean cross-similarity matrix using given threshold 'kappa'.\n\n"
"Use default parameter values for best results while computing cross-similarity using the binarize method.\n\n"
"The input feature arrays should be in the shape (x, y), where 'x' is the number of frames.\n\n"
"An exception is also thrown if either one of the input audio feature arrays are empty or if the output similarity matrix is empty.\n\n"
"References:\n"
"[1] Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification.New Journal of Physics.\n\n");


void CrossSimilarityMatrix::configure() {
  // configure parameters
  _frameStackStride = parameter("frameStackStride").toInt();
  _frameStackSize = parameter("frameStackSize").toInt();
  _kappa = parameter("kappa").toReal();
  _toBinary = parameter("toBinary").toBool();
}

// Construct a 'stacked-frames' feature vector from an input audio feature vector by given 'frameStackSize' and 'frameStackStride'
std::vector<std::vector<Real> > CrossSimilarityMatrix::stackFrames(std::vector<std::vector<Real> >& frames, int frameStackSize, int frameStackStride) const {

  if (frameStackSize == 1) {
    return frames;
  }
  size_t stopIdx;
  int increment = frameStackSize * frameStackStride;
  std::vector<std::vector<Real> > stackedFrames;
  stackedFrames.reserve(frames.size() - increment);
  std::vector<Real> stack;
  for (size_t i=0; i<(frames.size() - increment); i+=frameStackStride) {
    stopIdx = i + increment;
    for (size_t startTime=i; startTime<stopIdx; startTime+=frameStackStride) {
      stack.insert(stack.end(), frames[startTime].begin(), frames[startTime].end());
    }
    stackedFrames.push_back(stack);
    stack.clear();
  }
  return stackedFrames;
}


void CrossSimilarityMatrix::compute() {
  // get inputs and output
  std::vector<std::vector<Real> > queryFeature = _queryFeature.get();
  std::vector<std::vector<Real> > referenceFeature = _referenceFeature.get();
  std::vector<std::vector<Real> >& csm = _csm.get();

  if (queryFeature.empty())
    throw EssentiaException("CrossSimilarityMatrix: input queryFeature array is empty.");
  if (referenceFeature.empty())
    throw EssentiaException("CrossSimilarityMatrix: input referenceFeature array is empty.");

  // construct time embedding from input chroma features
  std::vector<std::vector<Real> >  queryFeatureStack = stackFrames(queryFeature, _frameStackSize, _frameStackStride);
  std::vector<std::vector<Real> >  referenceFeatureStack = stackFrames(referenceFeature, _frameStackSize, _frameStackStride);

  std::vector<std::vector<Real> > pdistances;
  // pairwise euclidean distance
  pdistances = pairwiseDistance(queryFeatureStack, referenceFeatureStack);
  if (pdistances.empty())
    throw EssentiaException("CrossSimilarityMatrix: empty array found inside euclidean cross similarity matrix.");

  // check whether to binarize the euclidean cross-similarity matrix using the given thresholds 
  if (_toBinary == true) {
    std::vector<std::vector<Real> > tpDistances = transpose(pdistances);
    size_t xRows = pdistances.size();
    size_t xCols = pdistances[0].size();
    size_t yRows = tpDistances.size();
    size_t yCols = tpDistances[0].size();

    std::vector<std::vector<Real> > similarityX;
    similarityX.assign(xRows, std::vector<Real>(xCols));
    // construct thresholded similarity matrix on axis X
    for (size_t k=0; k<xRows; k++) {
      for (size_t l=0; l<xCols; l++) {
        similarityX[k][l] = percentile(pdistances[k], _kappa*100) - pdistances[k][l];
      }
    }
    // binarise the array with heavisideStepFunction
    heavisideStepFunction(similarityX);
    std::vector<std::vector<Real> > similarityY(yRows, std::vector<Real>(yCols));
    // construct thresholded similarity matrix on axis Y
    for (size_t u=0; u<yRows; u++) {
      for (size_t v=0; v<yCols; v++) {
        similarityY[u][v] = percentile(tpDistances[u], _kappa*100) - tpDistances[u][v];
      }
    }
    // here we binarise and transpose the similarityY array same time in order to avoid redundant looping
    std::vector<std::vector<Real> > tSimilarityY(yCols, std::vector<Real>(yRows));
    for (size_t i=0; i<yRows; i++) {
      for (size_t j=0; j<yCols; j++) {
        if (similarityY[i][j] < 0) {
          tSimilarityY[j][i] = 0;
        }
        else if (similarityY[i][j] >= 0) {
          tSimilarityY[j][i] = 1;
        }
      }
    }
    // finally we construct out cross similarity matrix by multiplying similarityX and similarityY
    // [TODO]: replace TNT array matmult with Boost matrix in future for faster computation.
    TNT::Array2D<Real> simX = vecvecToArray2D(similarityX);
    TNT::Array2D<Real> simY = vecvecToArray2D(tSimilarityY);
    TNT::Array2D<Real> csmOut = TNT::operator*(simX, simY);
    csm = array2DToVecvec(csmOut);
  }
  // Use default cross-similarity computation method based on euclidean distances
  else {
    // returns pairwise euclidean distance
    csm = pdistances;
  }
}

} // namespace standard
} // namespace essentia
