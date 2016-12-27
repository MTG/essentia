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

#include "gfcc.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* GFCC::name = "GFCC";
const char* GFCC::category = "Spectral";
const char* GFCC::description = DOC("This algorithm computes the Gammatone-frequency cepstral coefficients of a spectrum. This is an equivalent of MFCCs, but using a gammatone filterbank (ERBBands) scaled on an Equivalent Rectangular Bandwidth (ERB) scale.\n"
"\n"
"References:\n"
"  [1] Y. Shao, Z. Jin, D. Wang, and S. Srinivasan, \"An auditory-based feature\n"
"  for robust speech recognition,\" in IEEE International Conference on\n"
"  Acoustics, Speech, and Signal Processing (ICASSP’09), 2009,\n"
"  pp. 4625-4628.");

void GFCC::configure() {
  _gtFilter->configure(INHERIT("inputSize"),
                       INHERIT("sampleRate"),
                       INHERIT("numberBands"),
                       INHERIT("lowFrequencyBound"),
                       INHERIT("highFrequencyBound"),
                       INHERIT("type"));
  _dct->configure("inputSize", parameter("numberBands"),
                  "outputSize", parameter("numberCoefficients"),
                  INHERIT("dctType"));
  _logbands.resize(parameter("numberBands").toInt());
  setCompressor(parameter("logType").toString());
}

void GFCC::compute() {
  // get the inputs and outputs
  const vector<Real>& spectrum = _spectrum.get();
  vector<Real>& gfcc = _gfcc.get();
  vector<Real>& bands = _bands.get();

  // filter the spectrum using a gammatone filterbank
  _gtFilter->input("spectrum").set(spectrum);
  _gtFilter->output("bands").set(bands);
  _gtFilter->compute();




  for (int i=0; i<int(bands.size()); ++i) {
    _logbands[i] = (*_compressor)(bands[i]);
  }

  // compute the DCT of these bands
  _dct->input("array").set(_logbands);
  _dct->output("dct").set(gfcc);
  _dct->compute();
}

void GFCC::setCompressor(std::string logType){
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
    throw EssentiaException("GFCC: Bad 'logType' parameter");
  }

}
