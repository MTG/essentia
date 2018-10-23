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
const char* Histogram::description = DOC("This algorithm computes a histogram. All values outside histogram range are ignored");

void Histogram::configure() {
  _normalize = parameter("normalize").toString(); 
  _minValue = parameter("minValue").toReal();
  _maxValue = parameter("maxValue").toReal();
  _numberBins = parameter("numberBins").toInt();

  if(_maxValue < _minValue)
    throw EssentiaException("Histogram: maxValue must be > minValue");

  binWidth =  (_maxValue - _minValue)/(Real)_numberBins;

  tempBinCenters.resize(_numberBins);

  tempBinCenters[0] = _minValue + (Real)(binWidth/2.0);
  for(int i = 1; i < _numberBins; i++) {
    tempBinCenters[i] = tempBinCenters[i-1] + binWidth;
  }

}

void Histogram::compute() {
  
  const std::vector<Real>& array = _array.get();
  std::vector<Real>& histogram = _histogram.get();
  std::vector<Real>& binCenters = _binCenters.get();
  
  histogram.resize(_numberBins);
  binCenters.assign(tempBinCenters.begin(), tempBinCenters.end());

  for(int i = 0; i < array.size(); i++){
    if(array[i] < _maxValue && array[i] >= _minValue)
      histogram[floor(array[i]/(Real)binWidth)]++;
    else if(array[i] == _maxValue)
      histogram[_numberBins-1]++;
  }

}
