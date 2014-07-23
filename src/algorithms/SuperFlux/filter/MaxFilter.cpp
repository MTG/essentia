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
const char* MaxFilter::description = DOC("Maximum filter for 1d signal (van Herk/Gil-Werman Algorithm ) "
"");


void MaxFilter::configure() {
 	
 	
 	//width has to be odd
    _width = parameter("width").toInt();
 _causal = parameter("Causal").toBool();

}

#ifdef HERKGIL

void MaxFilter::compute() {


  	const vector<Real>& array = _array.get();
  	
	vector<Real>& filtered = _filtered.get();

   	int size= array.size();

	filtered.resize(size);
	Real maxs = 0;
	
	Real cur_diff = 0;

// for herk gil algo s represent the whole width
if(_width%2==0)_width++;
int kl=(_width-1)/2;

vector<Real> cs(_width-2);
vector<Real> ds(_width-2);

// fill begining
maxs=array[0];
filtered[0]=maxs;
for (int i = 1 ; i < kl ; i++){
filtered[i]=max(maxs,array[i]);
}

for(int u = kl ; u<size ; u+=_width-1){
		ds[0]=array[u];
		
		for (int i=1;i<=_width-2;i++){
			ds[i] = max(ds[i-1],array[u+i]);
		}
		cs[_width-2] = array[u-1];
		
		for (int i = 1 ; i <= _width-2 ; i++){
			cs[_width-i-2] = max(cs[_width-i-1],array[u-i-1]);
		}
		
		for (int i = 0 ; i <= _width-2 ; i++){
			// filtered[u-kl+i] = max(cs[i],ds[i]);
			filtered[u+i] = max(cs[i],ds[i]);
		}
		
	
}

}



#else

void MaxFilter::compute() {
  	const vector<Real>& array = _array.get();
  	
	vector<Real>& filtered = _filtered.get();


  	int size= array.size();
	
	if(_width>=size)throw EssentiaException("recieved signal is smaller or equal than width");
	
	filtered.resize(size);


	Real cur_diff = 0;

	// if centered width represent half window
	if(!_causal){
	if(_width%2==0)_width++;
	_width=(_width-1)/2;
	}
	Real maxs=array[0];
	filtered[0]=maxs;
	for (int i = 1 ; i < _width ; i++){
	filtered[i]=max(maxs,array[i]);
	maxs=filtered[i];
	}
	

	for(int j = _width ; j<size ; j++){
		// if the outgoing term is not last max the new max is faster to compute
		
		int wmax = _causal?j:min(j+_width,size);
		if(j>_width && array[j-_width-1]<maxs){
			maxs = max(maxs,array[wmax]);
		}	
		else{
			
			maxs =array[j-_width];
			for (int k = j-_width+1 ; k<=wmax ; k++){
				maxs = max(maxs,array[k]);
			}
		}	
		filtered[j]=maxs;

	}
	

}


#endif



void MaxFilter::reset() {
  Algorithm::reset();

}



} // namespace standard

namespace streaming {


  AlgorithmStatus MaxFilter::process(){
  bool producedData = false;


	AlgorithmStatus status = acquireData();
	if (status != OK) {

	 
	  // acquireData() returns SYNC_OK if we could reserve both inputs and outputs
	  // being here means that there is either not enough input to process,
	  // or that the output buffer is full, in which cases we need to return from here
	  // cout << "peaks no fed" << endl;
	  return status;
	}


	std::vector<Real> arr = _array.tokens();
 	Real* dst = (Real*)_filtered.getFirstToken() ;
 	
 	*dst = *std::max_element(arr.begin(),arr.end());

	// give back the tokens that were reserved
	releaseData();
	

	return OK;
  
  }
}

} // namespace essentia





