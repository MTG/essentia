/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "dct.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* DCT::name = "DCT";
const char* DCT::description = DOC("This algorithm computes the Discrete Cosine Transform of an array.\n"
"It uses the DCT-II form, with the 1/sqrt(2) scaling factor for the first coefficient.\n"
"Note: The 'inputSize' parameter is only used as an optimization when the algorithm is configured. "
"The DCT will automatically adjust to the size of any input.\n"
"\n"
"References:\n"
"  [1] Discrete cosine transform - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Discrete_cosine_transform");


void DCT::configure() {
  int inputSize = parameter("inputSize").toInt();
  _outputSize = parameter("outputSize").toInt();

  createDctTable(inputSize, _outputSize);
}

void DCT::createDctTable(int inputSize, int outputSize) {
  // simple implementation using matrix multiplication, can probably be sped up
  // using a library like FFTW, for instance.
  if (outputSize > inputSize) {
    throw EssentiaException("DCT: 'outputSize' is greater than 'inputSize'. You can only compute the DCT with an output size smaller than the input size (i.e. you can only compress information)");
  }

  _dctTable = vector<vector<Real> >(outputSize, vector<Real>(inputSize));

  // scale for index = 0
  Real scale0 = 1.0 / sqrt(Real(inputSize));

  // scale for index != 0
  Real scale1 = Real(sqrt(2.0/inputSize));

  for (int i=0; i<outputSize; ++i) {
    Real scale = (i==0)? scale0 : scale1;

    Real freqMultiplier = Real(M_PI / inputSize * i);

    for (int j=0; j<inputSize; ++j) {
      _dctTable[i][j] = (Real)(scale * cos( freqMultiplier * ((Real)j + 0.5) ));
    }
  }
}


void DCT::compute() {

  const vector<Real>& array = _array.get();
  vector<Real>& dct = _dct.get();
  int inputSize = int(array.size());

  if (inputSize == 0) {
    throw EssentiaException("DCT: input array cannot be of size 0");
  }

  if (_dctTable.empty() ||
      inputSize != int(_dctTable[0].size()) ||
      _outputSize != int(_dctTable.size())) {
    createDctTable(inputSize, _outputSize);
  }

  dct.resize(_outputSize);

  for (int i=0; i<_outputSize; ++i) {
    dct[i] = 0.0;
    for (int j=0; j<inputSize; ++j) {
      dct[i] += array[j] * _dctTable[i][j];
    }
  }
}
