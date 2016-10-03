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

#include "superfluxnovelty.h"
#include "essentiamath.h"

namespace essentia {
namespace standard {
        
const char* SuperFluxNovelty::name = "SuperFluxNovelty";
const char* SuperFluxNovelty::category = "Rhythm";
const char* SuperFluxNovelty::description = DOC("Onset detection function for Superflux algorithm. See SuperFluxExtractor for more details.");
        
void SuperFluxNovelty::configure() {
  _binWidth = parameter("binWidth").toInt();
  _maxFilter->configure("width", _binWidth, "causal", false);
  _frameWidth = parameter("frameWidth").toInt();
}

void SuperFluxNovelty::compute() {
            
const vector< vector<Real> >& bands = _bands.get();
  Real& diffs = _diffs.get();
  
  int nFrames = bands.size();
  if(!nFrames) {
    throw EssentiaException("SuperFluxNovelty: empty frames");
  }

  int nBands= bands[0].size();
  if(!nBands){
    throw EssentiaException("SuperFluxNovelty: empty bands");
  }
            
  if (_frameWidth >= nFrames) {
    throw EssentiaException("SuperFluxNovelty: not enough frames for the specified frameWidth");
  }

  vector<Real> maxsBuffer(nBands, 0);
            
  // buffer for differences
  Real cur_diff;
  diffs = 0;
  for (int i=_frameWidth; i<nFrames; i++) {
    _maxFilter->input("signal").set(bands[i-_frameWidth]);
    _maxFilter->output("signal").set(maxsBuffer);
    _maxFilter->compute();
                
    cur_diff = 0;
                
    for (int j = 0;j<nBands;j++) {
      cur_diff= bands[i][j]-maxsBuffer[j];
      if (cur_diff > 0.0) {
        diffs +=cur_diff ;
      }
    }
  }
  return;
}
        
void SuperFluxNovelty::reset() {
  Algorithm::reset();
}

} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {
        
const char* SuperFluxNovelty::name = standard::SuperFluxNovelty::name;
const char* SuperFluxNovelty::category = standard::SuperFluxNovelty::category;
const char* SuperFluxNovelty::description = standard::SuperFluxNovelty::description;

AlgorithmStatus SuperFluxNovelty::process() {
  AlgorithmStatus status = acquireData();
  if (status != OK) {
    return status;
  }
  _algo->input("bands").set(_bands.tokens());
  _algo->output("differences").set(_diffs.firstToken());
  _algo->compute();
  
  // give back the tokens that were reserved
  releaseData();
  return OK;
}
        
} // namespace streaming
} // namespace essentia
