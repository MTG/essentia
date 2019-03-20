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

  // construct a new vector by stacking the input features by an specified 'frameStackStride' and 'frameStackSize'
  std::vector<std::vector<Real> >  queryFeatureStack = stackFrames(queryFeature, _frameStackSize, _frameStackStride);
  std::vector<std::vector<Real> >  referenceFeatureStack = stackFrames(referenceFeature, _frameStackSize, _frameStackStride);

  std::vector<std::vector<Real> > pdistances;
  // pairwise euclidean distance
  pdistances = pairwiseDistance(queryFeatureStack, referenceFeatureStack);

  // check whether to binarize the euclidean cross-similarity matrix using the given threshold kappa
  if (_toBinary == true) {
    std::vector<std::vector<Real> > tpDistances = transpose(pdistances);
    size_t queryRows = pdistances.size();
    size_t referenceRows = pdistances[0].size();

    std::vector<Real> thresholdX(queryRows);
    std::vector<Real> thresholdY(referenceRows);

    std::vector<std::vector<Real> > outputSimMatrix;
    outputSimMatrix.assign(queryRows, std::vector<Real>(referenceRows));
    // construct the binary output similarity matrix using the thresholds computed along the queryFeature axis
    for (size_t k=0; k<queryRows; k++) {
      thresholdX[k] = percentile(pdistances[k], _kappa*100);
      for (size_t l=0; l<referenceRows; l++) {
        if ((thresholdX[k] - pdistances[k][l]) > 0) {
          outputSimMatrix[k][l] = 1;
        }
        else if ((thresholdX[k] - pdistances[k][l]) <= 0) {
          outputSimMatrix[k][l] = 0.;
        }
      }

    }
    // update the binary output similarity matrix by multiplying with the thresholds computed along the referenceFeature axis
    for (size_t j=0; j<referenceRows; j++) {
      thresholdY[j] = percentile(tpDistances[j], _kappa*100);
      for (size_t i=0; i<queryRows; i++) {
        if ((thresholdY[j] - pdistances[i][j]) > 0) {
          outputSimMatrix[i][j] *= 1; 
        }
        else if ((thresholdY[j] - pdistances[i][j]) <= 0) {
          outputSimMatrix[i][j] *= 0;
        }
      }
    }
    csm = outputSimMatrix;
  }
  // Use default cross-similarity computation method based on euclidean distances
  else {
    // returns pairwise euclidean distance
    csm = pdistances;
  }
}

} // namespace standard
} // namespace essentia
