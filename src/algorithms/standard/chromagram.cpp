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
const char* Chromagram::description = DOC("This algorithm computes the Constant-Q chromagram using FFT. See ConstantQ algorithm for more details.\n");


void Chromagram::configure() {
  
  _binsPerOctave = parameter("binsPerOctave").toInt();
  int numberBins = parameter("numberBins").toInt();

  _octaves = numberBins /_binsPerOctave;  
  // TODO check if the configured numberBins matches a complete number of octaves
  // if not that throw a warning that only X first bins will be used 

  string normalizeType = parameter("normalizeType").toString();
  if (normalizeType == "none") _normalizeType = NormalizeNone;
  else if (normalizeType == "unit_sum") _normalizeType = NormalizeUnitSum;
  else if (normalizeType == "unit_max") _normalizeType = NormalizeUnitMax;
  else throw EssentiaException("Invalid normalize type for chromagram (none/unit_sum/unit_max): ", normalizeType);

  _spectrumCQ->configure(INHERIT("minFrequency"), INHERIT("numberBins"),
                         INHERIT("binsPerOctave"), INHERIT("sampleRate"),
                         INHERIT("threshold"), INHERIT("scale"),
                         INHERIT("windowType"), INHERIT("minimumKernelSize"),
                         INHERIT("zeroPhase"));
  _spectrumCQ->output("spectrumCQ").set(_chromaBuffer);
}

void Chromagram::compute() {

  const vector<Real>& signal = _signal.get();
  vector<Real>& chromagram = _chromagram.get();

  chromagram.assign(_binsPerOctave, 0.0);

  // Compute Constant-Q spectrogram
  _spectrumCQ->input("frame").set(signal);
  _spectrumCQ->compute();

  for (unsigned octave = 0; octave < _octaves; octave++) {
    unsigned firstBin = octave * _binsPerOctave;
    for (unsigned i = 0; i < _binsPerOctave; i++) {
        chromagram[i] += _chromaBuffer[firstBin+i];
    }
  }
  
  if (_normalizeType == NormalizeUnitSum) {
    normalizeSum(chromagram);
  }
  else if (_normalizeType == NormalizeUnitMax) {
    normalize(chromagram);
  }
}
