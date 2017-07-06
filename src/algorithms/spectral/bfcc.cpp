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

#include "bfcc.h"
#include "essentiamath.h" // lin2db

using namespace std;
using namespace essentia;
using namespace standard;

const char* BFCC::name = "BFCC";
const char* BFCC::category = "Spectral";
const char* BFCC::description = DOC("This algorithm computes the bark-frequency cepstrum coefficients of a spectrum. Bark bands and their subsequent usage in cepstral analysis have shown to be useful in percussive content [1, 2]\n"                                    
"This algorithm is implemented using the Bark scaling approach in the Rastamat version of the MFCC algorithm and in a similar manner to the MFCC-FB40 default specs:\n"
"\n"
"http://www.ee.columbia.edu/ln/rosa/matlab/rastamat/"
"\n"
"  - filterbank of 40 bands from 0 to 11000Hz\n"
"  - take the log value of the spectrum energy in each bark band\n"
"  - DCT of the 40 bands down to 13 mel coefficients\n"
"\n"
"The parameters of this algorithm can be configured in order to behave like Rastamat [3] as follows:\n"
"  - type = 'power' \n"
"  - weighting = 'linear'\n"
"  - lowFrequencyBound = 0\n"
"  - highFrequencyBound = 8000\n"
"  - numberBands = 26\n"
"  - numberCoefficients = 13\n"
"  - normalize = 'unit_max'\n"
"  - dctType = 3\n"
"  - logType = 'log'\n"
"  - liftering = 22\n"
"\n"
"In order to completely behave like Rastamat the audio signal has to be scaled by 2^15 before the processing and if the Windowing and FrameCutter algorithms are used they should also be configured as follows. \n"
"\n"
"FrameGenerator:\n"
"  - frameSize = 1102 \n"
"  - hopSize = 441 \n"
"  - startFromZero = True \n"
"  - validFrameThresholdRatio = 1 \n"
"\n"
"Windowing:\n"
"  - type = 'hann' \n"
"  - size = 1102 \n"
"  - zeroPadding = 946 \n"
"  - normalized = False \n"
"\n"
"This algorithm depends on the algorithms TriangularBarkBands (not the regular BarkBands algo as it is non-configurable) and DCT and therefore inherits their parameter restrictions. An exception is thrown if any of these restrictions are not met. The input \"spectrum\" is passed to the TriangularBarkBands algorithm and thus imposes TriangularBarkBands' input requirements. Exceptions are inherited by TriangualrBarkBands as well as by DCT.\n"
"\n"
"References:\n"
"  [1] P. Herrera, A. Dehamel, and F. Gouyon, \"Automatic labeling of unpitched percussion sounds in\n"
"  Audio Engineering Society 114th Convention, 2003,\n"
"  [2] W. Brent, \"Cepstral Analysis Tools for Percussive Timbre Identification in\n"
"  Proceedings of the 3rd International Pure Data Convention, Sao Paulo, Brazil, 2009,\n"
);

void BFCC::configure() {
  _triangularBarkFilter->configure(INHERIT("inputSize"),
                        INHERIT("sampleRate"),
                        INHERIT("numberBands"),
                        INHERIT("lowFrequencyBound"),
                        INHERIT("highFrequencyBound"),                        
                        INHERIT("weighting"),
                        INHERIT("normalize"),
                        INHERIT("type"));

  _dct->configure("inputSize", parameter("numberBands"),
                  "outputSize", parameter("numberCoefficients"),
                  INHERIT("dctType"),
                  INHERIT("liftering"));
  _logbands.resize(parameter("numberBands").toInt());

  setCompressor(parameter("logType").toString());

}

void BFCC::compute() {

  // get the inputs and outputs
  const vector<Real>& spectrum = _spectrum.get();
  vector<Real>& bfcc = _bfcc.get();
  vector<Real>& bands = _bands.get();

  // filter the spectrum using a mel-scaled filterbank
  _triangularBarkFilter->input("spectrum").set(spectrum);
  _triangularBarkFilter->output("bands").set(bands);
  _triangularBarkFilter->compute();

  // take the dB amplitude of the spectrum
  for (int i=0; i<int(bands.size()); ++i) {
    _logbands[i] = (*_compressor)(bands[i]);
  }

  // compute the DCT of these bands
  _dct->input("array").set(_logbands);
  _dct->output("dct").set(bfcc);
  _dct->compute();
}

void BFCC::setCompressor(std::string logType){
  if (logType == "natural"){
    _compressor = linear;
  }
  else if (logType == "dbpow"){
    _compressor = pow2db;
  }
  else if (logType == "dbamp"){
    _compressor = amp2db;
  }
  else if (logType == "log"){
    _compressor = log;
  }
  else{
    throw EssentiaException("BFCC: Bad 'logType' parameter");
  }

}
