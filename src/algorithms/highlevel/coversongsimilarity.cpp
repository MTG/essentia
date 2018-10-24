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
#include "coversongsimilarity.h"
#include "essentiamath.h"
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>

namespace essentia {
namespace standard {

const char* CoverSongSimilarity::name = "CoverSongSimilarity";
const char* CoverSongSimilarity::category = "Music similarity";
const char* CoverSongSimilarity::description = DOC("This algorithm computes a cover song similiarity measure from an input cross similarity matrix of two chroma vectors of a query and reference song using various alignment constraints of smith-waterman local-alignment algorithm.\n\n"
"This algorithm expects to recieve the input matrix from CrossSimilarityMatrix algorithm\n\n"
"The algorithm provides two different allignment contraints for computing the smith-waterman score matrix (check references).\n\n"
"Exceptions are thrown if the input similarity is not binary or empty.\n\n"
"References:\n"
"[1] Smith-Waterman algorithm Wikipedia (https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm).\n\n"
"[2] Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification.New Journal of Physics.\n\n"
"[3] Chen, N., Li, W., & Xiao, H. (2017). Fusing similarity functions for cover song identification. Multimedia Tools and Applications.\n");


void CoverSongSimilarity::configure() {
  _gammaO = parameter("gammaO").toReal();
  _gammaE = parameter("gammaE").toReal();

  std::string simType = toLower(parameter("simType").toString());
  if      (simType == "qmax") _simType = QMAX;
  else if (simType == "dmax") _simType = DMAX;
  else throw EssentiaException("CoverSongSimilarity: Invalid cover similarity type: ", simType);
}

void CoverSongSimilarity::compute() {
  // get input and output
  const std::vector<std::vector<Real> > simMatrix = _inputArray.get();
  std::vector<std::vector<Real> >& scoreMatrix = _scoreMatrix.get();

  if (simMatrix.empty())
      throw EssentiaException("CoverSongSimilarity: Input similarity matrix is empty");

  size_t Ny = simMatrix.size();
  size_t Nx = simMatrix[0].size();
  std::vector<std::vector<Real> > cumMatrix(Ny, std::vector<Real>(Nx, 0));

  Real c1 = 0;
  Real c2 = 0;
  Real c3 = 0;
  Real c4 = 0;
  Real c5 = 0;
  
  if (_simType == QMAX) {
    // iterate through the similarity matrix to recursively construct the qmax scoring cumilative matrix
    for(size_t i = 2; i < simMatrix.size(); i++) {
      for(size_t j = 2; j < simMatrix[i].size(); j++) {
        // measure the diagonal when a similarity is found in the input matrix
        if (simMatrix[i][j] == 1) {
          c1 = cumMatrix[i-1][j-1];
          c2 = cumMatrix[i-2][j-1];
          c3 = cumMatrix[i-1][j-2];
          Real row[3] = {c1, c2 , c3};
          cumMatrix[i][j] = *std::max_element(row, row+3) + 1;
          }
        else {
        // apply gap penalty onset for disruption and extension when similarity is not found in the input matrix
          c1 = cumMatrix[i-1][j-1] - gammaState(simMatrix[i-1][j-1], _gammaO, _gammaE);
          c2 = cumMatrix[i-2][j-1] - gammaState(simMatrix[i-2][j-1], _gammaO, _gammaE);
          c3 = cumMatrix[i-1][j-2] - gammaState(simMatrix[i-1][j-2], _gammaO, _gammaE);
          Real row2[4] = {0, c1, c2, c3};
          cumMatrix[i][j] = *std::max_element(row2, row2+4);
          }
      }
    }
    scoreMatrix = cumMatrix;
  }
  else if (_simType == DMAX) {
    // iterate through the similarity matrix to recursively construct the dmax scoring cumilative matrix
    for(size_t i = 2; i < simMatrix.size(); ++i) {
      for(size_t j = 2; i < simMatrix[i].size(); ++j) {
        // measure the diagonal when a similarity is found in the input matrix
        if (simMatrix[i][j] == 1.) {
          c2 = cumMatrix[i-2][j-1] + simMatrix[i-1][j];
          c3 = cumMatrix[i-1][j-2] + simMatrix[i][j-1];
          c4 = cumMatrix[i-3][j-1] + simMatrix[i-2][j] + scoreMatrix[i-1][j];
          c5 = cumMatrix[i-1][j-3] + simMatrix[i][j-2] + scoreMatrix[i][j-1];
          Real row[5] = {cumMatrix[i-1][j-1], c2, c3, c4, c5};
          cumMatrix[i][j] = *std::max_element(row, row+5) + 1;
        }
        else {
          // apply gap penalty onset for disruption and extension when similarity is not found in the input matrix
          c1 = cumMatrix[i-1][j-1] - gammaState(simMatrix[i-1][j-1], _gammaO, _gammaE);
          c2 = (cumMatrix[i-2][j-1] + simMatrix[i-1][j]) - gammaState(simMatrix[i-2][j-1], _gammaO, _gammaE);
          c3 = (cumMatrix[i-1][j-2] + simMatrix[i][j-1]) - gammaState(simMatrix[i-1][j-2], _gammaO, _gammaE);
          c4 = (cumMatrix[i-3][j-1] + simMatrix[i-2][j] + simMatrix[i-1][j]) - gammaState(simMatrix[i-3][j-1], _gammaO, _gammaE);
          c5 = (cumMatrix[i-1][j-3] + simMatrix[i][j-2] + simMatrix[i][j-1]) - gammaState(simMatrix[i-1][j-3], _gammaO, _gammaE);
          Real row2[6] = {0, c1, c2, c3, c4, c5};
          cumMatrix[i][j] = *std::max_element(row2, row2+6);
        }
      }
    }
    scoreMatrix = cumMatrix;
  }
}

// apply gap penalty for disruption and extension
Real CoverSongSimilarity::gammaState(Real value, const Real gammaO, const Real gammaE) const {
  if      (value == 1.) return gammaO;
  else if (value == 0.) return gammaE;
  else throw EssentiaException("CoverSongSimilarity:Non-binary elements found in the inputsimilarity matrix. Expected a binary similarity matrix!");
}

} // namespace standard
} // namespace essentia
