/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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

#include "MaxFilter.h"
#include "essentiamath.h"
//#define HERKGIL
//TODO:Validate and adapt for onlinemode HERKGIL

namespace essentia {
namespace standard {

const char* MaxFilter::name = "MaxFilter";
const char* MaxFilter::description = DOC("Maximum filter for 1d signal (van Herk/Gil-Werman Algorithm).\n"
"\n"
"References:\n"
"  TODO\n");

void MaxFilter::configure() {
    //width has to be odd
    _width = parameter("width").toInt();
    _causal = parameter("causal").toBool();
    _filledBuffer = false;


    // offset by width/2 if not causal as it's only a shift of output indexes

    _halfWidth = _width;
    if (_halfWidth % 2==0) _halfWidth++;
    _halfWidth = (_halfWidth-1) / 2;


    _bufferFillIdx = _causal?0:_halfWidth;
    
    
}



void MaxFilter::compute() {
  const vector<Real>& array = _array.get();
  vector<Real>& filtered = _filtered.get();

  int size = array.size();
    if(size<1)
        throw EssentiaException("MaxFilter has recieved an empty vector");
    
    filtered.resize(size);
    
    // local read index in buffer
    int readIdx = 0;

    // initially fill buffer
    if(!_filledBuffer){
        
        if(_bufferFillIdx == _causal?0:_halfWidth){
            _curMax = array[0];
            // we initiate the value here because we need to pad with an array value ( especially in non causal mode where )
            _buffer.resize(_width, _curMax);
        }
        
        int maxIdx =min(size,_width-_bufferFillIdx);
        for(int i =0 ;i<maxIdx ; i++){
            _buffer[ _bufferFillIdx] = array[readIdx];
            _curMax =max(array[readIdx],_curMax);
            filtered[i] = _curMax;
            readIdx++;
            _bufferFillIdx++;
    }
        
        
        _filledBuffer =  _bufferFillIdx==_width;
    }
  
    // fill and compute max of the curent circular buffer
    for(int j = readIdx ; j < size ; j++){
        _bufferFillIdx %=_width;
        _buffer[_bufferFillIdx] = array[j];
        filtered[j] = *std::max_element(_buffer.begin(), _buffer.end());
        _bufferFillIdx++;
    }
}

    

void MaxFilter::reset() {
  Algorithm::reset();
    _buffer.clear();
    _filledBuffer = false;
    _bufferFillIdx = 0;
}

} // namespace standard
} // namespace essentia

//// we may wamt to implement this later, if dealing with big vectors
//    

