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

#include "idct.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* IDCT::name = "IDCT";
const char* IDCT::category = "Standard";
const char* IDCT::description = DOC("This algorithm computes the Inverse Discrete Cosine Transform of an array.\n"
"It can be configured to perform the inverse DCT-II form, with the 1/sqrt(2) scaling factor for the first coefficient or the inverse DCT-III form based on the HTK implementation.\n"
"\n"
"IDCT can be used to compute smoothed Mel Bands. In order to do this:\n"
"  - compute MFCC\n"
"  - smoothedMelBands = 10^(IDCT(MFCC)/20)\n"
"Note: The second step assumes that 'logType' = 'dbamp' was used to compute MFCCs, otherwise that formula should be changed in order to be consistent.\n"
"\n"
"Note: The 'inputSize' parameter is only used as an optimization when the algorithm is configured. "
"The IDCT will automatically adjust to the size of any input.\n"
"\n"
"References:\n"
"  [1] Discrete cosine transform - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Discrete_cosine_transform \n"
"  [2] HTK book, chapter 5.6 ,\n"
"  http://speech.ee.ntu.edu.tw/homework/DSP_HW2-1/htkbook.pdf");


void IDCT::configure() {
  int inputSize = parameter("inputSize").toInt();
  _outputSize = parameter("outputSize").toInt();
  _type = parameter("dctType").toInt();
  _lifter = parameter("liftering").toInt();
  if (_type == 2){
    createIDctTableII(inputSize, _outputSize);
  }
  else if (_type == 3){
    createIDctTableIII(inputSize, _outputSize);
  }
  else {
    throw EssentiaException("IDCT: Bad DCT type.");
  }
}

void IDCT::createIDctTableII(int inputSize, int outputSize) {
  // simple implementation using matrix multiplication, can probably be sped up
  // using a library like FFTW, for instance.
  if (outputSize < inputSize) {
    throw EssentiaException("IDCT: 'outputSize' is smaller than 'inputSize'. You can only compute the IDCT with an output size greater or equal than the input size");
  }

  _idctTable = vector<vector<Real> >(outputSize, vector<Real>(inputSize));

  // scale for index = 0
  Real scale0 = 1.0 / sqrt(Real(outputSize));

  // scale for index != 0
  Real scale1 = Real(sqrt(2.0/outputSize));

  for (int i=0; i<inputSize; ++i) {
    Real scale = (i==0)? scale0 : scale1;

    Real freqMultiplier = Real(M_PI / outputSize * i);

    for (int j=0; j<outputSize; ++j) {
      _idctTable[j][i] = (Real)(scale * cos( freqMultiplier * ((Real)j + 0.5) ));
    }
  }
}

void IDCT::createIDctTableIII(int inputSize, int outputSize) {
  // simple implementation using matrix multiplication, can probably be sped up
  // using a library like FFTW, for instance.
  if (outputSize < inputSize) {
    throw EssentiaException("IDCT: 'outputSize' is smaller than 'inputSize'. You can only compute the IDCT with an output size greater or equal than the input size");
  }

  _idctTable = vector<vector<Real> >(outputSize, vector<Real>(inputSize));
  // This implementation is used instead of the referenced in order to match the behaviour of the HTK
  // http://speech.ee.ntu.edu.tw/homework/DSP_HW2-1/htkbook.pdf

   Real scaleDefault = Real(sqrt(2.0/outputSize));

   for (int i=0; i<inputSize; ++i) {
     Real freqMultiplier = Real(M_PI / outputSize * i);
     Real scale = (i == 0) ? scaleDefault/2 : scaleDefault;
     for (int j=0; j<outputSize; ++j) {
       _idctTable[j][i] = (Real)(scale * cos( freqMultiplier * ( (Real)j + 0.5) ) );

     }
   }
}


void IDCT::compute() {

  const vector<Real>& input = _dct.get();
  vector<Real>& idct = _idct.get();

  vector<Real> dct = input;

  int inputSize = int(input.size());

  if (inputSize == 0) {
    throw EssentiaException("IDCT: input array cannot be of size 0");
  }

  if (_idctTable.empty() ||
      inputSize != int(_idctTable[0].size()) ||
      _outputSize != int(_idctTable.size())) {
    if (_type == 2){
      createIDctTableII(inputSize, _outputSize);
    }
    else if (_type == 3){
      createIDctTableIII(inputSize, _outputSize);
    }
    else {
      throw EssentiaException("Bad DCT type.");
    }
  }

  idct.resize(_outputSize);

  //Inverse liftering
  if (_lifter != 0.0){
    for (int i=1; i< inputSize; ++i) {
        dct[i] /= 1.0  + (_lifter / 2 ) * sin( (M_PI * i) / (double)_lifter );
    }
  }

  for (int j=0; j<_outputSize; ++j) {
    idct[j] = 0.0;
    for (int i=0; i<inputSize; ++i) {
      idct[j] += dct[i] * _idctTable[j][i];
    }
  }

}
