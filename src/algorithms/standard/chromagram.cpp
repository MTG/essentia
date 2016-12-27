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

#include "chromagram.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Chromagram::name = "Chromagram";
const char* Chromagram::category = "Tonal";
const char* Chromagram::description = DOC("This algorithm computes the chromagram of the Constant Q Transform.\n"
"\n");


void Chromagram::configure() {
  
  _sampleRate = parameter("sampleRate").toDouble();
  _minFrequency = parameter("minFrequency").toDouble();
  _maxFrequency = parameter("maxFrequency").toDouble();
  _binsPerOctave = parameter("binsPerOctave").toInt();
  _threshold = parameter("threshold").toDouble();

  string normalizeType = parameter("normalizeType").toString();
  if (normalizeType == "none") _normalizeType = NormalizeNone;
  else if (normalizeType == "unit_sum") _normalizeType = NormalizeUnitSum;
  else if (normalizeType == "unit_max") _normalizeType = NormalizeUnitMax;
  else throw EssentiaException("Invalid normalize type for chromagram (none/unit_sum/unit_max): ", normalizeType);

  unsigned int uK = (unsigned int) ceil(_binsPerOctave * log(_maxFrequency/_minFrequency)/log(2.0)); 
  _octaves = (int)floor(double(uK/_binsPerOctave))-1;  

  _constantq->configure("minFrequency", _minFrequency,
                        "maxFrequency", _maxFrequency,
                        "binsPerOctave", _binsPerOctave,
                        "sampleRate", _sampleRate,
                        "threshold", _threshold);

  _constantq->output("constantq").set(_CQBuffer);
  _magnitude->input("complex").set(_CQBuffer);
}

void Chromagram::compute() {

  const vector<complex<Real> >& signal = _signal.get();
  vector<Real>& chromagram = _chromagram.get();

  chromagram.assign(_binsPerOctave, 0.0);

  _constantq->input("frame").set(signal);
  _constantq->compute();
  
  _magnitude->output("magnitude").set(_ChromaBuffer);
  _magnitude->compute();

  for (unsigned octave=0; octave <= _octaves; octave++) {
    unsigned firstBin = octave*_binsPerOctave;
    for (unsigned i = 0; i < _binsPerOctave; i++) {
        chromagram[i] += _ChromaBuffer[firstBin+i];
    }
  }
  
  if (_normalizeType == NormalizeUnitSum) {
    normalizeSum(chromagram);
  }
  else if (_normalizeType == NormalizeUnitMax) {
    normalize(chromagram);
  }
}
