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

#include "Triangularbands.h"
#include "essentiamath.h"

namespace essentia{
namespace standard{

const char* Triangularbands::name = "Triangularbands";
const char* Triangularbands::description = DOC("This algorithm computes the energy of an input spectrum for an arbitrary number of overlapping Triangular frequency bands. For each band the power-spectrum (mag-squared) is summed.\n"
"\n"
"Parameter \"Triangularbands\" must contain at least 2 frequencies, they all must be positive and must be ordered ascentdantly, otherwise an exception will be thrown. Triangularbands is only defined for spectra, which size is greater than 1.\n"
"\n"
);

void Triangularbands::configure() {
  _bandFrequencies = parameter("frequencyBands").toVectorReal();
  _sampleRate = parameter("sampleRate").toReal();
  if ( _bandFrequencies.size() < 2 ) {throw EssentiaException("Triangularbands: the 'Triangularbands' parameter contains only one element (i.e. two elements are required to construct a band)");}
  for (int i = 1; i < int(_bandFrequencies.size()); ++i) {
    if ( _bandFrequencies[i] < 0 ) {throw EssentiaException("Triangularbands: the 'Triangularbands' parameter contains a negative value");}
    if (_bandFrequencies[i-1] >= _bandFrequencies[i] ) {throw EssentiaException("Triangularbands: the values in the 'Triangularbands' parameter are not in ascending order or there exists a duplicate value");}
  }
  _isLog = parameter("Log").toBool();
}




void Triangularbands::compute() {
  const std::vector<Real>& spectrum = _spectrumInput.get();
  std::vector<Real>& bands = _bandsOutput.get();

  if (spectrum.size() <= 1) {throw EssentiaException("Triangularbands: the size of the input spectrum is not greater than one");}

  Real frequencyscale = (_sampleRate / 2.0) / (spectrum.size() - 1);
  int nBands = int(_bandFrequencies.size() - 2);

  bands.resize(nBands);
  std::fill(bands.begin(), bands.end(), (Real) 0.0);

  for (int i=0; i<nBands; i++) {

    int startBin = int(_bandFrequencies[i] / frequencyscale +.5);
    int midBin = int(_bandFrequencies[i + 1] / frequencyscale +.5 );
    int endBin = int(_bandFrequencies[i + 2] / frequencyscale +.5);
  

	// finished
    if (startBin >= int(spectrum.size())) {break;}
	// going to far
    if (endBin > int(spectrum.size())) {endBin = spectrum.size();}
	
	//Compute normalization factor
	Real norm=0;
	if(midBin!=startBin && midBin!= endBin && endBin!=startBin)for (int j=startBin; j<=endBin; j++) {norm+=  j<midBin? (j-startBin)/(midBin - startBin) : 1-(j-midBin)/(endBin-midBin);}
    
    for (int j=startBin; j<=endBin; j++) {
    Real TriangF;
    if(midBin!=startBin && midBin!= endBin && endBin!=startBin){
    	TriangF = j<midBin? (j-startBin)/(midBin - startBin) : 1-(j-midBin)/(endBin-midBin);
    	TriangF/=norm;
	}
	// case of single bin band
	else if (startBin== endBin){
		TriangF = 1;
	}
	//double bin band
	else{
		TriangF = 0.5;
	}
	
	
      bands[i] += TriangF * spectrum[j] * spectrum[j]; 
    }
    if(_isLog){bands[i] = log10(1+bands[i]);}
  }

}



}// namespace standard
}// namespace essentia

