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

#include "SuperFluxNovelty.h"
#include "essentiamath.h"



namespace essentia {
namespace standard {


const char* SuperFluxNovelty::name = "SuperFluxNovelty";
const char* SuperFluxNovelty::description = DOC("Novelty curve in Superflux algorithm : Maximum filter and differentiation for onset detection robust again vibrato"
"Input : filterbank like spectrogram");


void SuperFluxNovelty::configure() {
 	
 	
 	
    _binW = parameter("binWidth").toInt();
    //if(_binW%2==0)_binW++;
    _maxf->configure("width",_binW,"Causal",false);

    
	_frameWi = parameter("frameWidth").toInt();

	_online = parameter("Online").toBool();


}


void SuperFluxNovelty::compute() {

  	const vector< vector<Real> >& bands = _bands.get();
  	
	vector<Real>& diffs = _diffs.get();

  int nFrames = bands.size();
  if(!nFrames){
  throw EssentiaException("SuperFluxNovelty : empty frames");
  }  
  int nBands= bands[0].size();
  if(!nBands){
  throw EssentiaException("SuperFluxNovelty : empty bands ");
  }
  E_DEBUG( EAlgorithm, "got " << nFrames <<"frames, for frameWidth" << _frameWi << "with " << nBands<<" bands");
  if(_frameWi>=nFrames){
  
  throw EssentiaException("SuperFluxNovelty : no enough frames comparing to frame witdh");
  }
  
  
// ONLINE MODE all results are advanced by frame width

  if (_online){diffs.resize(nFrames-_frameWi);	}
  else { diffs.resize(nFrames);}

  int onlinestep = _online?_frameWi:0;	

  vector<Real> maxsBuffer(nBands,0);


  Real cur_diff;

  for (int i = _frameWi ; i< nFrames;i++){

	diffs[i-onlinestep]=0;
	_maxf->input("signal").set(bands[i-_frameWi]);
	_maxf->output("signal").set(maxsBuffer);
	_maxf->compute();
	
	cur_diff = 0;
	//cout<<bands[i][15]<<"//"<<bands[i-_frameWi][15]<<endl;
	for (int j = 0;j<nBands;j++){
	
		cur_diff= bands[i][j]-maxsBuffer[j];
		if(cur_diff>0.0){diffs[i-onlinestep] +=cur_diff ; }

		
	}


}
return;
}







void SuperFluxNovelty::reset() {
  Algorithm::reset();

}


// TODO in the case of lower accuracy in evaluation
// implement post-processing steps for methods in OnsetDetection, which required it
// wrapping the OnsetDetection algo
// - smoothing?
// - etc., whatever was requiered in original matlab implementations

} // namespace standard
} // namespace essentia








// 



namespace essentia {
namespace streaming {

const char* SuperFluxNovelty::name = standard::SuperFluxNovelty::name;
const char* SuperFluxNovelty::description = standard::SuperFluxNovelty::description;



AlgorithmStatus SuperFluxNovelty::process() {
  bool producedData = false;


    AlgorithmStatus status = acquireData();
    if (status != OK) {
      // acquireData() returns SYNC_OK if we could reserve both inputs and outputs
      // being here means that there is either not enough input to process,
      // or that the output buffer is full, in which cases we need to return from here
      return status;
    }
    

    _algo->input("bands").set(_bands.tokens());
    _algo->output("Differences").set(_diffs.tokens());
	
    _algo->compute();

    // give back the tokens that were reserved
    releaseData();

    return OK;
  
}


} // namespace streaming
} // namespace essentia
