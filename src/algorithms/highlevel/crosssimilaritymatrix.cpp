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
const char* CrossSimilarityMatrix::category = "Music similarity";
const char* CrossSimilarityMatrix::description = DOC("This algorithm compute a cross similarity matrix from two audio feature vectors of a query and reference song as proposed in [1].\n" "\n"
" ------------------ \n"
"\n"
"[1]. Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification.New Journal of Physics.\n"
"\n");


void CrossSimilarityMatrix::configure() {
  _tau = parameter("tau").toInt(); 
  _m = parameter("m").toInt();
  _kappa = parameter("kappa").toDouble(); 
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

    // check whether to transpose by oti
    if (_oti == true) {
        int otiIdx = optimalTranspositionIndex(queryFeature, referenceFeature, _noti);
        std::rotate(referenceFeature.begin(), referenceFeature.end() - otiIdx, referenceFeature.end());
        //rotateByIndex(referenceFeature, otiIdx);
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
       throw EssentiaException("empty array found while calculating euclidian similarity");

    // transposing the array of pairwsie distance
    std::vector<std::vector<Real> > tDistances = transpose(pdistances);

    // ephisilon
    std::vector<std::vector<Real> > ephX(pdistances.size());
    std::vector<std::vector<Real> > ephY(tDistances.size());

    std::vector<Real> tempXrow;
    for (size_t i=0; i<pdistances.size(); i++) {
        tempXrow.push_back(percentile(pdistances[i], _kappa));
        ephX[i] = tempXrow;
        tempXrow.clear();
    }

    std::vector<Real> tempYrow;
    for (size_t j=0; j<tDistances.size(); j++) {
        tempYrow.push_back(percentile(tDistances[j], _kappa));
        ephY[j] = tempYrow;
        tempYrow.clear();
    }

    std::vector<std::vector<Real> > similarityX(pdistances.size());
    std::vector<std::vector<Real> > similarityY(tDistances.size());

    for (size_t k=0; k<pdistances.size(); k++) {
        std::transform(ephX[k].begin(),ephX[k].end(),pdistances[k].begin(),std::back_inserter(similarityX[k]),std::minus<Real>());
    }

    for (size_t l=0; l<tDistances.size(); l++) {
        std::transform(ephX[l].begin(),ephX[l].end(),tDistances[l].begin(),std::back_inserter(similarityY[l]),std::minus<Real>());
    }

    // Finally we construct out cross similarity matrix by doing dot product of the similarity matrices
    for (size_t x=0; x<similarityX.size(); x++) {
        for (size_t y=0; y<similarityY.size(); y++) {
            csm[x][y] = dotProduct(similarityX[x], similarityY[y]);
        }
    }
}


// Construct a timespace delay embedding from an input audio feature vector as described in [1].
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


// computes global averaged chroma hpcp as described in [2]
std::vector<Real> CrossSimilarityMatrix::globalAverageChroma(std::vector<std::vector<Real> >& inputFeature) const {

    std::vector<Real> globalChroma;
    // tranpose the array from (time_axis, chroma_bin) to (chroma_bin, time-axis)for easy calculation
    inputFeature = transpose(inputFeature);

    // sum all the value along the time axis of chroma hpcp feature
    for (size_t i=0; i<inputFeature.size(); i++) {
        globalChroma.push_back(std::accumulate(inputFeature[i].begin(), inputFeature[i].end(), 0));
    }     
    
    double maxItem = *std::max_element(globalChroma.begin(), globalChroma.end());
    // divide the sum array by the max element to normalise it to 0-1 range
    for (size_t j=0; j<globalChroma.size(); j++) {
        globalChroma[j] = globalChroma[j] / maxItem;
    }
    return globalChroma;
}


// Compute the optimal transposition index for transposing reference song feature to the musical key of query song feature as described in [2].
int CrossSimilarityMatrix::optimalTranspositionIndex(std::vector<std::vector<Real> >& featureA, std::vector<std::vector<Real> >& featureB, int nshifts) const {

    std::vector<Real> globalChromaA = globalAverageChroma(featureA);
    std::vector<Real> globalChromaB = globalAverageChroma(featureB);
    std::vector<Real> chromaBcopy = globalChromaB;
    std::vector<Real> valueAtShifts;

    for(int i=1; i<=nshifts; i++) {
        // circular rotate the input globalchroma by an index 'i'
        std::rotate(globalChromaB.begin(), globalChromaB.end() - i, globalChromaB.end());
        //rotateByIndex(globalChromaB, i);
        // compute the dot product of the query global chroma and the shifted global chroma of reference song and append to an array
        valueAtShifts.push_back(dotProduct(globalChromaA, globalChromaB));
        globalChromaB = chromaBcopy;
    }
    // compute the optimal index by finding the index of maximum element in the array of value at various shifts
    int maxValue = argmax(valueAtShifts);
    return maxValue;
}





} // namespace standard
} // namespace essentia

