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


double gammaState(double value, float gammaO=0.5, float gammaE=0.5);
double arrayMax(const float* arr, size_t length);

namespace essentia {
namespace standard {

const char* CoverSongSimilarity::name = "CoverSongSimilarity";
const char* CoverSongSimilarity::category = "Music similarity";
const char* CoverSongSimilarity::description = DOC("This algorithm computes a cover song similiarity measure from an input cross recurrent plot \
of two chroma vectors of a query and reference song using various alignment constraints of smith-waterman local-alignment algorithm.\n" "\n"
" ------------------ \n"
"\n"
"[1].Smith-Waterman algorithm Wikipedia (https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm).\n"
"\n"
"[2]. Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification.New Journal of Physics.\n"
"\n"
"[3]. Chen, N., Li, W., & Xiao, H. (2017). Fusing similarity functions for cover song identification. Multimedia Tools and Applications.\n");


void CoverSongSimilarity::configure() {
  _gammaO = parameter("gammaO").toFloat();
  _gammaE = parameter("gammaE").toFloat();

  std::string simType = toLower(parameter("simType").toString());
  if      (simType == "qmax") _simType = QMAX;
  else if (simType == "dmax") _simType = DMAX;
  else throw EssentiaException("Invalid cover similarity type: ", simType);
}

void CoverSongSimilarity::compute() {

    // get inputs and output
    const std::vector<std::vector<Real> > simMatrix = _inputArray.get();
    std::vector<std::vector<Real> >& scoreMatrix = _scoreMatrix.get();
    //Real similarityMeasure = _similarityMeasure.get();

    size_t Nx = simMatrix[0].size();
    size_t Ny = simMatrix.size();
    std::vector<std::vector<Real> > cumMatrix(Ny, std::vector<Real>(Nx, 0));

    float c1 = 0;
    float c2 = 0;
    float c3 = 0;
    float c4 = 0;
    float c5 = 0;

    if (_simType == QMAX)
    {
        //iterate through the similarity matrix to recursively construct the qmax scoring cumilative matrix
        for(size_t i = 2; i < simMatrix.size(); i++){
            for(size_t j = 2; j < simMatrix[i].size(); j++){
                //measure the diagonal when a similarity is found in the input matrix
                if (simMatrix[i][j] == 1){
                    c1 = cumMatrix[i-1][j-1];
                    c2 = cumMatrix[i-2][j-1];
                    c3 = cumMatrix[i-1][j-2];
                    float row[3] = {c1, c2 , c3};
                    cumMatrix[i][j] =  arrayMax(row, 3) + 1;
                }
                else
                {
                    // apply gap penalty onset for disurption and extension when similarity is not found in the input matrix
                    c1 = cumMatrix[i-1][j-1] - gammaState(simMatrix[i-1][j-1], _gammaO, _gammaE);
                    c2 = cumMatrix[i-2][j-1] - gammaState(simMatrix[i-2][j-1], _gammaO, _gammaE);
                    c3 = cumMatrix[i-1][j-2] - gammaState(simMatrix[i-1][j-2], _gammaO, _gammaE);
                    float row2[4] = {0, c1, c2, c3};
                    cumMatrix[i][j] = arrayMax(row2, 4);
                }
            }
        }
        scoreMatrix = cumMatrix;
    }
    else if (_simType == DMAX)
    {
        //iterate through the similarity matrix to recursively construct the dmax scoring cumilative matrix
        for(size_t i = 2; i < simMatrix.size(); ++i){
            for(size_t j = 2; i < simMatrix[i].size(); ++j){

                // measure the diagonal when a similarity is found in the input matrix
                if (simMatrix[i][j] == 1.){
                    c2 = cumMatrix[i-2][j-1] + simMatrix[i-1][j];
                    c3 = cumMatrix[i-1][j-2] + simMatrix[i][j-1];
                    c4 = cumMatrix[i-3][j-1] + simMatrix[i-2][j] + scoreMatrix[i-1][j];
                    c5 = cumMatrix[i-1][j-3] + simMatrix[i][j-2] + scoreMatrix[i][j-1];
                    float row[5] = {cumMatrix[i-1][j-1], c2, c3, c4, c5};
                    cumMatrix[i][j] =  arrayMax(row, 5) + 1;
                }
                else
                {
                    // apply gap penalty onset for disurption and extension when similarity is not found in the input matrix
                    c1 = cumMatrix[i-1][j-1] - gammaState(simMatrix[i-1][j-1]);
                    c2 = (cumMatrix[i-2][j-1] + simMatrix[i-1][j]) - gammaState(simMatrix[i-2][j-1], _gammaO, _gammaE);
                    c3 = (cumMatrix[i-1][j-2] + simMatrix[i][j-1]) - gammaState(simMatrix[i-1][j-2], _gammaO, _gammaE);
                    c4 = (cumMatrix[i-3][j-1] + simMatrix[i-2][j] + simMatrix[i-1][j]) - gammaState(simMatrix[i-3][j-1], _gammaO, _gammaE);
                    c5 = (cumMatrix[i-1][j-3] + simMatrix[i][j-2] + simMatrix[i][j-1]) - gammaState(simMatrix[i-1][j-3], _gammaO, _gammaE);
                    float row2[6] = {0, c1, c2, c3, c4, c5};
                    cumMatrix[i][j] = arrayMax(row2, 6);
                }
            }
        }
        scoreMatrix = cumMatrix;
    }
}


} //namespace standard
} //namespace essentia


//apply gap penalty for disurption and extension
double gammaState(double value, float gammaO, float gammaE)
{
    if (value == 1.){
        return gammaO;
    }
    else if (value == 0.){
        return gammaE;
    }
    else{
        //std::cout << "\nNon-binary elements found in the binary similarity matrix " << value << std::endl;
        return 0;
    }
}


// find max element in an array
double arrayMax(const float* arr, size_t length)
{

    double maximum = arr[0];
    for (size_t i=1; i< length; i++) {
        if (maximum < arr[i]){
            maximum = arr[i];
        }
    }
    return maximum;
}
