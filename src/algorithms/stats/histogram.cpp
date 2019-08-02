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

using namespace essentia;
using namespace standard;

const char* Histogram::name = "Histogram";
const char* Histogram::category = "Statistics";
const char* Histogram::description = DOC("This algorithm computes a histogram. Values outside the range are ignored");

void Histogram::configure() {
  _normalize = parameter("normalize").toString(); 
  _minValue = parameter("minValue").toReal();
  _maxValue = parameter("maxValue").toReal();
  _numberBins = parameter("numberBins").toInt();

  if(_maxValue < _minValue)
    throw EssentiaException("Histogram: maxValue must be > minValue");

  if(_maxValue == _minValue) {
    if(_numberBins > 1)
       throw EssentiaException("Histogram: numberBins must = 1 when maxValue = minValue");
  }

  binWidth =  (_maxValue - _minValue)/(Real)_numberBins;

  tempBinEdges.resize(_numberBins+1);

  tempBinEdges[0] = _minValue;
  for(std::vector<Real>::iterator it = tempBinEdges.begin()+1; it != tempBinEdges.end(); it++) {
    *it = *(it-1) + binWidth;
  }

}

void Histogram::compute() {
  
  const std::vector<Real>& array = _array.get();
  std::vector<Real>& histogram = _histogram.get();
  std::vector<Real>& binEdges = _binEdges.get();
  
  histogram.resize(_numberBins);
  binEdges.assign(tempBinEdges.begin(), tempBinEdges.end());

  for(size_t i = 0; i < array.size(); i++){
    if(array[i] < _maxValue && array[i] >= _minValue)
      histogram[floor(array[i]/(Real)binWidth)]++;
    else if(array[i] == _maxValue) 
      histogram[_numberBins-1]++;
  }

  if(_normalize != "none"){
    Real denominator = 0;
    if(_normalize == "unit_sum") {
      for(std::vector<Real>::iterator it = histogram.begin(); it != histogram.end(); it++) {
        denominator += *it;
      }
    }
    if(_normalize == "unit_max") {
      for(std::vector<Real>::iterator it = histogram.begin(); it != histogram.end(); it++) {
        if(*it > denominator)
          denominator = *it;
      }
    }
    for(std::vector<Real>::iterator it = histogram.begin(); it != histogram.end(); it++) { 
      *it = *it/denominator;    
    }
  }
}
