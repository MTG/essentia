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

#include "histogram.h"
#include "essentiamath.h"
#include "vector"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Histogram::name = "Histogram";
const char* Histogram::category = "Statistics";
const char* Histogram::description = DOC("This algorithm computes the histogram from an input array of real values");

void Histogram::configure() {
  //_minRange.resize(parameter("minRange").toInt());
  //_maxRange.resize(parameter("maxRange").toInt());
}

void Histogram::compute() {

  // get the inputs and outputs
  const vector<Real>& inputArray = _inputArray.get();
  <vector<Real>>& histogramArray = _histogramArray.get();
  const unsigned int numBins = _numBins.get();
  string normMode = _normMode.get();
  int minRange = _minRange.get();
  int maxRange = _maxRange.get();

  // normalze the array with the chosen mode
  if (normMode == "unit_sum"){
    unitSumNorm(&inputArray);
  }
  else if (normMode == "unit_max") {
    unitMaxNorm(&inputArray, maxRange);
  }
  else if (normMode == "None") {
  }
  else{
    throw EssentiaException("Histogram: invalid 'normMode' parameter");
  }

  double minValue = *min_element(inputArray.begin(), inputArray.end());
  double maxValue = *max_element(inputArray.begin(), inputArray.end());

  vector<double> binArray(numBins);
  vector<double> binFreq(numBins);
  double binSize = (minValue + maxValue) / numBins;
  double valueIterator = minValue;
  int freq=0;
  int j=0;
  for(int i=0; i<inputArray.size(); i++){
      vector<double> myRow;
      if (i==0){
          binArray[i] = minValue;
          myRow.push_back(minValue);
      }
      else{
          binArray[i] = valueIterator + binSize;
          myRow.push_back(valueIterator + binSize);
          freq = 0;
          for(j=0; j<inputArray.size(); j++){
              if (inputArray[j]>=valueIterator && inputArray[j]<=binArray[i]){
                  freq ++;
              }
          }
      }
      myRow.push_back(freq);
      binFreq[i] = freq;
      valueIterator = binArray[i];
      histogramArray.push_back(myRow);
    }
    histogramArray.resize(numBins);
}


// compute unit_max normalization of an array
void Histogram::unitMaxNorm(vector<Real>& inputArray, double maxRange){
    double maxValue = *max_element(inputArray.begin(), inputArray.end());
    for(int i=0; i<inputArray.size(); i++){
        inputArray[i] = (inputArray[i]*maxRange) / maxValue;
    }
}


// compute unit_sum normalization of an array
void Histogram::unitSumNorm(vector<Real>& inputArray){
  double normFactor = inputArray.norm(); //using essentiamath l2norm functions
  for(int i=0; i<inputArray.size(); i++){
      inputArray[i] = inputArray[i] / normFactor;
  }
}
