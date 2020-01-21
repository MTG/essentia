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
const char* CrossSimilarityMatrix::description = DOC("This algorithm computes a euclidean cross-similarity matrix of two sequences of frame features. Similarity values can be optionally binarized\n\n"
"The default parameters for binarizing are optimized according to [1] for cover song identification using chroma features. \n\n"
"The input feature arrays are vectors of frames of features in the shape (n_frames, n_features), where 'n_frames' is the number frames, 'n_features' is the number of frame features.\n\n"
"An exception is also thrown if either one of the input feature arrays are empty or if the output similarity matrix is empty.\n\n"
"References:\n"
"[1] Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification. New Journal of Physics.\n\n");


void CrossSimilarityMatrix::configure() {
  // configure parameters
  _frameStackStride = parameter("frameStackStride").toInt();
  _frameStackSize = parameter("frameStackSize").toInt();
  _binarizePercentile = parameter("binarizePercentile").toReal();
  _binarize = parameter("binarize").toBool();
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
  stack.reserve(frames[0].size() * frameStackSize);
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

  // construct a new vector by stacking the input features by an specified 'frameStackStride' and 'frameStackSize'
  std::vector<std::vector<Real> >  queryFeatureStack = stackFrames(queryFeature, _frameStackSize, _frameStackStride);
  std::vector<std::vector<Real> >  referenceFeatureStack = stackFrames(referenceFeature, _frameStackSize, _frameStackStride);

  // check whether to binarize the euclidean cross-similarity matrix using the given threshold kappa
  if (_binarize) {
    // pairwise euclidean distance
    std::vector<std::vector<Real> > pdistances = pairwiseDistance(queryFeatureStack, referenceFeatureStack);
    size_t queryFeatureSize = pdistances.size();
    size_t referenceFeatureSize = pdistances[0].size();

    std::vector<Real> thresholdQuery(queryFeatureSize);
    std::vector<Real> thresholdReference(referenceFeatureSize);

    csm.assign(queryFeatureSize, std::vector<Real>(referenceFeatureSize, 1));
    // construct the binary output similarity matrix using the thresholds computed along the queryFeature axis
    for (size_t k=0; k<queryFeatureSize; k++) {
      thresholdQuery[k] = percentile(pdistances[k], _binarizePercentile*100);
      for (size_t l=0; l<referenceFeatureSize; l++) {
        if (pdistances[k][l] > thresholdQuery[k]) {
          csm[k][l] = 0;
        }
      }
    }
    // update the binary output similarity matrix by multiplying with the thresholds computed along the referenceFeature axis
    for (size_t j=0; j<referenceFeatureSize; j++) {
      _status = true;
      for (size_t i=0; i<queryFeatureSize; i++) {
        if (_status) thresholdReference[j] = percentile(getColsAtVecIndex(pdistances, j), _binarizePercentile * 100);
        if (pdistances[i][j] > thresholdReference[j]) {
          csm[i][j] = 0;
        }
        _status = false;
      }
    }
  }
  // Use default cross-similarity computation method based on euclidean distances
  else {
    // returns pairwise euclidean distance
    csm = pairwiseDistance(queryFeatureStack, referenceFeatureStack);
  }
}

// returns a column corresponding to a specified index in the given 2D input matrix
std::vector<Real> CrossSimilarityMatrix::getColsAtVecIndex(std::vector<std::vector<Real> >& inputMatrix, int index) const {
  
  std::vector<Real> cols;
  cols.reserve(inputMatrix.size());
  for (size_t i=0; i<inputMatrix.size(); i++) {
    cols.push_back(inputMatrix[i][index]);
  }
  return cols;
}

} // namespace standard
} // namespace essentia
