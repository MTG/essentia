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


//TODO: create a real streaming mode and not standard mode hack...
#include "SuperFluxPeaks.h"
#include <complex>
#include <limits>
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {


const char* SuperFluxPeaks::name = "SuperFluxPeaks";
const char* SuperFluxPeaks::description = DOC("Peak peaking from Superflux algorithm (see SuperFluxExtractor for references)");


void SuperFluxPeaks::configure() {
	frameRate = parameter("frameRate").toReal();
	_pre_avg = int(frameRate* parameter("pre_avg").toReal() / 1000.);
	_pre_max = int(frameRate * parameter("pre_max").toReal() / 1000.);
	
	// convert to seconds
	_combine = parameter("combine").toReal()/1000.;

// 	_threshold = parameter("threshold").toReal();
	
	_rawMode = parameter("rawmode").toBool();
	_startZero = parameter("startFromZero").toBool();

	_movAvg->configure("size",_pre_avg);
	_maxf->configure("width",_pre_max,"Causal",true);
    
    _threshold = parameter("threshold").toReal();
	
	lastPidx = -1;
    peakTime = 0;

}


void SuperFluxPeaks::compute() {
// RT parameters





  	const vector<Real>& signal = _signal.get();
	vector<Real>& peaks = _peaks.get();
  	if (signal.empty()) {
		peaks.resize(0);
    	return;
  	}

	int size = signal.size();


	vector<Real> avg(size);
	_movAvg->reset();
	_movAvg->input("signal").set(signal);
	_movAvg->output("signal").set(avg);
	_movAvg->compute();


	vector<Real> maxs(size);
    _maxf->reset();
	_maxf->input("signal").set(signal);
	_maxf->output("signal").set(maxs);
	_maxf->compute();




	int nDetec=0;
	int minIdx = max(_pre_avg,_pre_max)-1 ;
	for( int i =minIdx; i < size;i++){
        
		if(signal[i]==maxs[i] && signal[i]>avg[i]+_threshold && signal[i]>0){

			peakTime = i*1.0/frameRate;
			if((nDetec>0 && peakTime-peaks[nDetec-1]>_combine)  ||  nDetec ==0) {
				peaks[nDetec] = peakTime;
				nDetec++;
			
			}
		}
		
		
	}

peaks.resize(nDetec);




return;
 
}






} // namespace standard
} // namespace essentia



#include "algorithmfactory.h"

namespace essentia {
namespace streaming {

const char* SuperFluxPeaks::name = standard::SuperFluxPeaks::name;
const char* SuperFluxPeaks::description = standard::SuperFluxPeaks::description;


    void SuperFluxPeaks::consume() {
        current_t+=1.0/framerate;
        std::vector<Real> out = std::vector<Real>(aqs);
        	_algo->input("novelty").set(_signal.tokens());
        	_algo->output("peaks").set(out);
        	_algo->compute();
        if(out.size()>0 && out[out.size()-1]>0 && ((onsTime.size()>0 && current_t-onsTime.back()>_combine )|| onsTime.size()==0) ){
            onsTime.push_back(current_t+aqs/framerate);
        }
        
        
    }
    
    void SuperFluxPeaks::finalProduce() {
        _peaks.push((std::vector<Real>) onsTime);
        
        current_t = 0;
        reset();
    }
    
    
    void SuperFluxPeaks::reset(){
        current_t=0;
        onsTime.clear();
    }

} // namespace streaming
} // namespace essentia
