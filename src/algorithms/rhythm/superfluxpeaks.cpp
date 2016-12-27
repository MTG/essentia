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

#include "superfluxpeaks.h"
#include <complex>
#include <limits>
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {
    
const char* SuperFluxPeaks::name = "SuperFluxPeaks";
const char* SuperFluxPeaks::category = "Rhythm";
const char* SuperFluxPeaks::description = DOC("This algorithm detects peaks of an onset detection function computed by the SuperFluxNovelty algorithm. See SuperFluxExtractor for more details.");


void SuperFluxPeaks::configure() {
  frameRate = parameter("frameRate").toReal();
  
  // convert to frame number
  _pre_avg = int(frameRate* parameter("pre_avg").toReal() / 1000.);
  _pre_max = int(frameRate * parameter("pre_max").toReal() / 1000.);
  
  if(_pre_avg <= 1) {
    throw EssentiaException("SuperFluxPeaks: too small _pre_averaging filter size");
  }
  if(_pre_max<=1) {
    throw EssentiaException("SuperFluxPeaks: too small _pre_maximum filter size");
  }
  
  // convert to seconds
  _combine = parameter("combine").toReal()/1000.;
  
  _movAvg->configure("size",_pre_avg);
  _maxf->configure("width",_pre_max,"causal",true);
  
  _threshold = parameter("threshold").toReal();
  _ratioThreshold = parameter("ratioThreshold").toReal();
  
  _startPeakTime = 0;
  nDetec=0;
}


void SuperFluxPeaks::compute() {
  
  const vector<Real>& signal = _signal.get();
  vector<Real>& peaks = _peaks.get();
  if (signal.empty()) {
    peaks.resize(0);
    return;
  }
  
  int size = signal.size();
  
  vector<Real> avg(size);
  _movAvg->input("signal").set(signal);
  _movAvg->output("signal").set(avg);
  _movAvg->compute();
  
  vector<Real> maxs(size);
  _maxf->input("signal").set(signal);
  _maxf->output("signal").set(maxs);
  _maxf->compute();

  int localnDetec = 0;
  for( int i=0; i<=size; i++) {
    // we want to avoid ratioThreshold noisy activation in really low flux parts so we set noise floor
    // to 10-7 by default (REALLY LOW for a flux)
    if(signal[i]==maxs[i] && signal[i]>1e-8) {
      bool isOverLinearThreshold = _threshold > 0 && signal[i] > avg[i]+_threshold;
      bool isOverratioThreshold = _ratioThreshold > 0 && avg[i] > 0 && signal[i]/avg[i] > _ratioThreshold;
    
      if(isOverLinearThreshold || isOverratioThreshold) {
        Real peakTime = _startPeakTime + i*1.0/frameRate;
        if((localnDetec > 0 && peakTime-peaks[localnDetec-1] > _combine) || localnDetec == 0) {
          peaks[localnDetec] = peakTime;
          nDetec++;
		  localnDetec++;
        }
      }
    } 
  }
  _startPeakTime += size*1.0/frameRate;

  peaks.resize(localnDetec);
}


} // namespace standard
} // namespace essentia


#include "algorithmfactory.h"

namespace essentia {
namespace streaming {

const char* SuperFluxPeaks::name = standard::SuperFluxPeaks::name;
const char* SuperFluxPeaks::category = standard::SuperFluxPeaks::category;
const char* SuperFluxPeaks::description = standard::SuperFluxPeaks::description;

void SuperFluxPeaks::consume() {

  int _aquireSize = _signal.acquireSize();
  std::vector<Real> out = std::vector<Real>(_aquireSize);

  _algo->input("novelty").set(_signal.tokens());
  _algo->output("peaks").set(out);
  _algo->compute();

  if (out.size() > 0) {
    // trim firstpart if needed
    bool trimBeg = false;
    if(onsetTimes.size() > 0 && (current_t + out[0] - onsetTimes.back() < _combine)) {
      trimBeg = true;
    }
    
    // copy if there is something to copy
    if (!trimBeg || onsetTimes.size()>1) {
      onsetTimes.insert(onsetTimes.end(), out.begin(), out.end()-(trimBeg ? 1 : 0));
    }
  }
  current_t += _aquireSize / framerate;
}


void SuperFluxPeaks::finalProduce() {
  _peaks.push((std::vector<Real>) onsetTimes);
  current_t = 0;
  reset();
}


void SuperFluxPeaks::reset(){
  current_t = 0;
  onsetTimes.clear();
  _algo->reset();
}

} // namespace streaming
} // namespace essentia
