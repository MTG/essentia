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

#include "coversongsimilarity.h"
#include "essentiamath.h"
#include "essentia/utils/tnt/tnt2vector.h"
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>

using namespace essentia;

Real maxElementArray(std::vector<std::vector<Real> >& inputMatrix);
Real gammaState(Real value, const Real disOnset, const Real disExtension);

namespace essentia {
namespace standard {

const char* CoverSongSimilarity::name = "CoverSongSimilarity";
const char* CoverSongSimilarity::category = "Music Similarity";
const char* CoverSongSimilarity::description = DOC("This algorithm computes a cover song similiarity measure from a binary cross similarity matrix input between two chroma vectors of a query and reference song using various alignment constraints of smith-waterman local-alignment algorithm.\n\n"
"This algorithm expects to recieve the binary similarity matrix input from essentia 'ChromaCrossSimilarity' algorithm or essentia 'CrossSimilarityMatrix' with parameter 'binarize=True'.\n\n"
"The algorithm provides two different allignment contraints for computing the smith-waterman score matrix (check references).\n\n"
"Exceptions are thrown if the input similarity matrix is not binary or empty.\n\n"
"References:\n\n"
"[1] Smith-Waterman algorithm (Wikipedia, https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm).\n\n"
"[2] Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification.New Journal of Physics.\n\n"
"[3] Chen, N., Li, W., & Xiao, H. (2017). Fusing similarity functions for cover song identification. Multimedia Tools and Applications.\n");


void CoverSongSimilarity::configure() {
  _disOnset = parameter("disOnset").toReal();
  _disExtension = parameter("disExtension").toReal();
  std::string distanceType = toLower(parameter("distanceType").toString());
  std::string simType = toLower(parameter("alignmentType").toString());
  if      (simType == "serra09") _simType = SERRA09;
  else if (simType == "chen17") _simType = CHEN17;
  else throw EssentiaException("CoverSongSimilarity: Invalid cover similarity type: ", simType);
  if      (distanceType == "symmetric") _distanceType = SYMMETRIC;
  else if (distanceType == "asymmetric") _distanceType = ASYMMETRIC;
  else throw EssentiaException("CoverSongSimilarity: Invalid distance type: ", simType);
}

void CoverSongSimilarity::compute() {
  // get input and output
  const std::vector<std::vector<Real> > simMatrix = _inputArray.get();
  std::vector<std::vector<Real> >& scoreMatrix = _scoreMatrix.get();
  Real& distance = _distance.get();

  if (simMatrix.empty())
      throw EssentiaException("CoverSongSimilarity: Input similarity matrix is empty");

  size_t xFrames = simMatrix.size();
  size_t yFrames = simMatrix[0].size();
  // assign the output scoreMatrix with zeros
  scoreMatrix.assign(xFrames, std::vector<Real>(yFrames, 0));
  Real c1 = 0;
  Real c2 = 0;
  Real c3 = 0;
  Real c4 = 0;
  Real c5 = 0;

  if (_simType == SERRA09) {
    // iterate through the similarity matrix to recursively construct the qmax scoring cumilative matrix
    for(size_t i = 2; i < xFrames; i++) {
      for(size_t j = 2; j < yFrames; j++) {
        // measure the diagonal when a similarity is found in the input matrix
        if (int(simMatrix[i][j]) == 1) {
          c1 = scoreMatrix[i-1][j-1];
          c2 = scoreMatrix[i-2][j-1];
          c3 = scoreMatrix[i-1][j-2];
          Real row[3] = {c1, c2 , c3};
          scoreMatrix[i][j] = *std::max_element(row, row+3) + 1;
        }
        else {
        // apply gap penalty onset for disruption and extension when similarity is not found in the input matrix
          c1 = scoreMatrix[i-1][j-1] - gammaState(simMatrix[i-1][j-1], _disOnset, _disExtension);
          c2 = scoreMatrix[i-2][j-1] - gammaState(simMatrix[i-2][j-1], _disOnset, _disExtension);
          c3 = scoreMatrix[i-1][j-2] - gammaState(simMatrix[i-1][j-2], _disOnset, _disExtension);
          Real row2[4] = {0, c1, c2, c3};
          scoreMatrix[i][j] = *std::max_element(row2, row2+4);
        }
      }
    }
  }
  else if (_simType == CHEN17) {
    // iterate through the similarity matrix to recursively construct the dmax scoring cumilative matrix
    for(size_t i = 3; i < xFrames; i++) {
      for(size_t j = 3; j < yFrames; j++) {
        // measure the diagonal when a similarity is found in the input matrix
        if (int(simMatrix[i][j]) == 1) {
          c2 = scoreMatrix[i-2][j-1] + simMatrix[i-1][j];
          c3 = scoreMatrix[i-1][j-2] + simMatrix[i][j-1];
          c4 = scoreMatrix[i-3][j-1] + simMatrix[i-2][j] + simMatrix[i-1][j];
          c5 = scoreMatrix[i-1][j-3] + simMatrix[i][j-2] + simMatrix[i][j-1];
          Real row[5] = {scoreMatrix[i-1][j-1], c2, c3, c4, c5};
          scoreMatrix[i][j] = *std::max_element(row, row+5) + 1;
        }
        else {
          // apply gap penalty onset for disruption and extension when similarity is not found in the input matrix
          c1 = scoreMatrix[i-1][j-1] - gammaState(simMatrix[i-1][j-1], _disOnset, _disExtension);
          c2 = (scoreMatrix[i-2][j-1] + simMatrix[i-1][j]) - gammaState(simMatrix[i-2][j-1], _disOnset, _disExtension);
          c3 = (scoreMatrix[i-1][j-2] + simMatrix[i][j-1]) - gammaState(simMatrix[i-1][j-2], _disOnset, _disExtension);
          c4 = (scoreMatrix[i-3][j-1] + simMatrix[i-2][j] + simMatrix[i-1][j]) - gammaState(simMatrix[i-3][j-1], _disOnset, _disExtension);
          c5 = (scoreMatrix[i-1][j-3] + simMatrix[i][j-2] + simMatrix[i][j-1]) - gammaState(simMatrix[i-1][j-3], _disOnset, _disExtension);
          Real row2[6] = {0, c1, c2, c3, c4, c5};
          scoreMatrix[i][j] = *std::max_element(row2, row2+6);
        }
      }
    }
  }
  if (_distanceType == SYMMETRIC) {
    distance = maxElementArray(scoreMatrix);
  }
  else if (_distanceType == ASYMMETRIC) {
    // compute cover song similarity distance by normalising it with the length of reference song as described in [2].
    distance = sqrt(yFrames) / maxElementArray(scoreMatrix);
  }
}

} // namespace standard
} // namespace essentia


#include "algorithmfactory.h"

namespace essentia {
namespace streaming {


const char* CoverSongSimilarity::name = standard::CoverSongSimilarity::name;
const char* CoverSongSimilarity::description = DOC("This algorithm computes a cover song similiarity measure from a binary cross similarity matrix input between two chroma vectors of a query and reference song using specific alignment constraint [2] of smith-waterman local-alignment algorithm [1].\n\n"
"This algorithm expects to recieve the binary similarity matrix input from essentia 'ChromaCrossSimilarity' algorithm or essentia 'CrossSimilarityMatrix' with parameter 'binarize=True'.\n\n"
"Exceptions are thrown if the input similarity matrix is not binary or empty.\n\n"
"References:\n\n"
"[1] Smith-Waterman algorithm (Wikipedia, https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm).\n\n"
"[2] Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification.New Journal of Physics.\n\n");

void CoverSongSimilarity::configure() {
  _disOnset = parameter("disOnset").toReal();
  _disExtension = parameter("disExtension").toReal();
  _pipeDistance = parameter("pipeDistance").toBool();
  std::string distanceType = toLower(parameter("distanceType").toString());
  if      (distanceType == "symmetric") _distanceType = SYMMETRIC;
  else if (distanceType == "asymmetric") _distanceType = ASYMMETRIC;
  else throw EssentiaException("CoverSongSimilarity: Invalid distance type: ", distanceType);
  _c1 = 0;
  _c2 = 0;
  _c3 = 0;
  // set the acquire and release size for the inputs and outputs for the streaming network
  input("inputArray").setAcquireSize(_minFrameAcquireSize);
  input("inputArray").setReleaseSize(_minFrameReleaseSize);
  output("scoreMatrix").setAcquireSize(1);
  output("scoreMatrix").setReleaseSize(1);
}

AlgorithmStatus CoverSongSimilarity::process() {

  const std::vector<std::vector<Real> >& inputFrames = _inputArray.tokens();
  std::vector<TNT::Array2D<Real> >& scoreMatrix = _scoreMatrix.tokens();
  std::vector<Real>& distance = _distance.tokens();

  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired (in: " << _inputArray.acquireSize()
             << " - out: " << _scoreMatrix.acquireSize() << ")");
  
  if (status != OK) {
    if (!shouldStop()) return status;

    // if shouldStop is true, that means there is no more audio coming, so we need
    // to take what's left to fill in half-frames, instead of waiting for more
    // data to come in (which would have done by returning from this function)
    int available = input("inputArray").available();
    if (available == 0) return NO_INPUT;

    input("inputArray").setAcquireSize(available);
    input("inputArray").setReleaseSize(available);

    return process();
  }

  std::vector<std::vector<Real> > inputFramesCopy = inputFrames; 
  /* if we have less input frame streams than the required '_minFrameSize' in the last stream, 
   we append the already acquired frames of the current stream until it satisfies the condition */
  if (input("inputArray").acquireSize() < _minFrameAcquireSize) {
    for (int i=0; i<(_minFrameAcquireSize - input("inputArray").acquireSize()); i++) {
      inputFramesCopy.push_back(inputFrames[i]);
    }
  }

  _xFrames = inputFramesCopy.size();
  _yFrames = inputFramesCopy[0].size();

  if (_iterIdx == 0) {
    _mainScoreMatrix.assign(_xFrames, std::vector<Real>(_yFrames, 0));
  }
  else {
    _mainScoreMatrix.push_back(std::vector<Real>(_yFrames, 0));
  }

  // compute qmax alignment score matrix for each 3 sub frames of input stream 
  subFrameQmax(inputFramesCopy);

  // compute distance
  if (_distanceType == SYMMETRIC) {
    distance[0] = maxElementArray(_mainScoreMatrix);
  }
  else if (_distanceType == ASYMMETRIC) {
    // compute cover song similarity distance by normalising it with the length of reference song as described in [2].
    distance[0] = sqrt(_yFrames) / maxElementArray(_mainScoreMatrix);
  }
  if (_pipeDistance) E_INFO(distance[0]);
  _iterIdx++;
  scoreMatrix[0] = vecvecToArray2D(_perFrameScoreMatrix);
  _perFrameScoreMatrix.clear();
  releaseData();
  return OK;
}

// compute smith-waterman alignment based on [2] for each 3 frames streams of array input
void CoverSongSimilarity::subFrameQmax(std::vector<std::vector<Real> >& inputFrames) {
  
  if (int(_xFrames) != _minFrameAcquireSize) throw EssentiaException("CoverSongSimilarity: Wrong input frame size!");
  int i = 2;
  for (size_t j=2; j<_yFrames; j++) {
    // measure the diagonal when a similarity is found in the input matrix
    if (int(inputFrames[i][j]) == 1) {
      _c1 = _mainScoreMatrix[_iterRow-1][j-1];
      _c2 = _mainScoreMatrix[_iterRow-2][j-1];
      _c3 = _mainScoreMatrix[_iterRow-1][j-2];
      Real row[3] = {_c1, _c2 , _c3};
      _mainScoreMatrix[_iterRow][j] = *std::max_element(row, row+3) + 1;
    }
    else {
    // apply gap penalty onset for disruption and extension when similarity is not found in the input matrix
    _c1 = _mainScoreMatrix[_iterRow-1][j-1] - gammaState(inputFrames[i-1][j-1], _disOnset, _disExtension);
    _c2 = _mainScoreMatrix[_iterRow-2][j-1] - gammaState(inputFrames[i-2][j-1], _disOnset, _disExtension);
    _c3 = _mainScoreMatrix[_iterRow-1][j-2] - gammaState(inputFrames[i-1][j-2], _disOnset, _disExtension);
    Real row2[4] = {0, _c1, _c2, _c3};
    _mainScoreMatrix[_iterRow][j] = *std::max_element(row2, row2+4);
    }
  }
  _perFrameScoreMatrix.push_back(_mainScoreMatrix[_iterRow]);
  _iterRow++;
};


} // namespace streaming
} // namespace essentia


// apply gap penalty for disruption  onset and extension
Real gammaState(Real value, const Real disOnset, const Real disExtension) {
  if      (int(value) == 1) return disOnset;
  else if (int(value) == 0) return disExtension;
  else throw EssentiaException("CoverSongSimilarity:Non-binary elements found in the input similarity matrix. Expected a binary similarity matrix!");
}


// returns the maximum element in the given 2D vector
Real maxElementArray(std::vector<std::vector<Real> >& inputMatrix) {

  Real maxElement = INT_MIN;
  for (size_t i=0; i<inputMatrix.size(); i++) {
    for (size_t j=0; j<inputMatrix[i].size(); j++) {
      if (inputMatrix[i][j] > maxElement) maxElement = inputMatrix[i][j];
    }
  }
  return maxElement;
}


