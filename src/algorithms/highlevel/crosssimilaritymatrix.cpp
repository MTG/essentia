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
const char* CrossSimilarityMatrix::description = DOC("This algorithm computes a binary cross similarity matrix from two chromagam feature vectors of a query and reference song.\n\n"
"Use HPCP algorithm for computing the chromagram for best results.\n\n"
"The input chromagram should be in the shape (x, numbins), where 'x' is number of frames and 'numbins' stands for number of bins in the chromagram. An exception isthrown otherwise.\n\n"
"An exception is also thrown if either one of the input audio feature arrays are empty or if the cross similarity matrix is empty.\n\n"
"References:\n"
"[1] Serra, J., Gómez, E., & Herrera, P. (2008). Transposing chroma representations to a common key, IEEE Conference on The Use of Symbols to Represent Music and Multimedia Objects.\n\n"
"[2] Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification.New Journal of Physics.\n\n"
"[3] Serra, Joan, et al. Chroma binary similarity and local alignment applied to cover song identification. IEEE Transactions on Audio, Speech, and Language Processing 16.6 (2008).\n");


void CrossSimilarityMatrix::configure() {
  _tau = parameter("tau").toInt();
  _m = parameter("m").toInt();
  _kappa = parameter("kappa").toReal();
  _noti = parameter("noti").toInt();
  _oti = parameter("oti").toBool();
  _toBlocked = parameter("toBlocked").toBool();
}

void CrossSimilarityMatrix::compute() {
  // get inputs and output
  std::vector<std::vector<Real> > queryFeature = _queryFeature.get();
  std::vector<std::vector<Real> > referenceFeature = _referenceFeature.get();
  std::vector<std::vector<Real> >& csm = _csm.get();
  std::vector<std::vector<Real> > pdistances;

  if (queryFeature.empty())
      throw EssentiaException("CrossSimilarityMatrix: input queryFeature array is empty.");

  if (referenceFeature.empty())
      throw EssentiaException("CrossSimilarityMatrix: input referenceFeature array is empty.");

  // check whether to transpose by oti
  if (_oti == true) {
      int otiIdx = optimalTranspositionIndex(queryFeature, referenceFeature, _noti);
      std::rotate(referenceFeature.begin(), referenceFeature.end() - otiIdx, referenceFeature.end());
  }

  // check if delay embedding needed
  if (_toBlocked == true) {
      // construct time embedding from input chroma features
      std::vector<std::vector<Real> >  timeEmbedA = toTimeEmbedding(queryFeature, _m, _tau);
      std::vector<std::vector<Real> >  timeEmbedB = toTimeEmbedding(referenceFeature, _m, _tau);
      // pairwise euclidean distance
      pdistances = pairwiseDistance(timeEmbedA, timeEmbedB);
  }
  else {
      // pairwise euclidean distance
      pdistances = pairwiseDistance(queryFeature, referenceFeature);
  }
  if (pdistances.empty())
      throw EssentiaException("CrossSimilarityMatrix: Empty array found on the euclidean cross similarity matrix.");

  // transposing the array of pairwsie distance
  std::vector<std::vector<Real> > tpDistances = transpose(pdistances);

  std::vector<std::vector<Real> > ephX(pdistances.size(), std::vector<Real>(1, 0));
  std::vector<std::vector<Real> > ephY(tpDistances.size(), std::vector<Real>(1, 0));
  for (size_t i=0; i<pdistances.size(); i++) {
      ephX[i][0] = percentile(pdistances[i], _kappa*100);
  }
  for (size_t j=0; j<tpDistances.size(); j++) {
      ephY[j][0] = percentile(tpDistances[j], _kappa*100);
  }

  std::vector<std::vector<Real> > similarityX(pdistances.size(), std::vector<Real>(pdistances[0].size(), 0));
  std::vector<std::vector<Real> > similarityY(tpDistances.size(), std::vector<Real>(tpDistances[0].size(), 0));
  // Construct thresholded similarity matrix on axis X
  for (size_t k=0; k<pdistances.size(); k++) {
      for (size_t l=0; l<pdistances[k].size(); l++) {
          similarityX[k][l] = ephX[k][0] - pdistances[k][l];
      }
  }
  // Construct thresholded similarity matrix on axis Y
  for (size_t u=0; u<tpDistances.size(); u++) {
      for (size_t v=0; v<tpDistances[u].size(); v++) {
          similarityY[u][v] = ephY[v][0] - tpDistances[u][v];
      }
  }

  // Binarise the array with heavisideStepFunction
  heavisideStepFunction(similarityX);
  heavisideStepFunction(similarityY);

  csm.assign(similarityX.size(), std::vector<Real>(similarityY.size(), 0));
  // tranpose similarityY vector for dot product
  similarityY = transpose(similarityY);

  // Finally we construct out cross similarity matrix by doing dot product
  for (size_t x=0; x<similarityX.size(); x++) {
      for (size_t y=0; y<similarityY.size(); y++) {
          csm[x][y] = dotProduct(similarityX[x], similarityY[y]);
      }
  }
}

// Construct a stacked chroma embedding from an input chroma audio feature vector 
// [TODO] In future use beat-synchronised stacked embeddings
std::vector<std::vector<Real> > CrossSimilarityMatrix::toTimeEmbedding(std::vector<std::vector<Real> >& inputArray, int m, int tau) const {

    int stopIdx;
    int increment = m*tau;
    int frameSize = inputArray[0].size() - increment;
    int yDim = inputArray.size() * m;
    std::vector<std::vector<Real> > timeEmbedding(frameSize, std::vector<Real>(yDim, 0));
    std::vector<Real> tempRow;
    tempRow.reserve(yDim*tau);

    for (int i=0; i<frameSize; i+=tau) {
        stopIdx = i + increment;
        for (int startTime=i; startTime<=stopIdx; startTime+=tau) {
            if (startTime == i) {
                tempRow = inputArray[startTime];
            }
            else {
                tempRow.insert(tempRow.end(), inputArray[startTime].begin(), inputArray[startTime].end());
            }
        timeEmbedding[i] = tempRow;
        };
    }
    return timeEmbedding;
}

// computes global averaged chroma hpcp as described in [1]
std::vector<Real> CrossSimilarityMatrix::globalAverageChroma(std::vector<std::vector<Real> >& inputFeature) const {

  size_t numbins = inputFeature[0].size();
  std::vector<Real> globalChroma(numbins);

  Real tSum;
  for (size_t j=0; j<numbins; j++) {
      tSum = 0;
      for (size_t i=0; i<inputFeature.size(); i++) {
          tSum += inputFeature[i][j];
      }
      globalChroma[j] = tSum;
  }
  // divide the sum array by the max element to normalise it to 0-1 range
  normalize(globalChroma);
  return globalChroma;
}

// Compute the optimal transposition index for transposing reference song feature to the musical key of query song feature as described in [1].
int CrossSimilarityMatrix::optimalTranspositionIndex(std::vector<std::vector<Real> >& chromaA, std::vector<std::vector<Real> >& chromaB, int nshifts) const {
    
    std::vector<Real> globalChromaA = globalAverageChroma(chromaA);
    std::vector<Real> globalChromaB = globalAverageChroma(chromaB);
    std::vector<Real> valueAtShifts;
    int prevIdx = 0;
    for(int i=1; i<=nshifts; i++) {
        // circular rotate the input globalchroma by an index 'i'
        std::rotate(globalChromaB.begin(), globalChromaB.end() - (i - prevIdx), globalChromaB.end());
        // compute the dot product of the query global chroma and the shifted global chroma of reference song and append to an array
        valueAtShifts.push_back(dotProduct(globalChromaA, globalChromaB));
        prevIdx++;
    }
    // compute the optimal index by finding the index of maximum element in the array of value at various shifts
    return argmax(valueAtShifts);
}

} // namespace standard
} // namespace essentia
