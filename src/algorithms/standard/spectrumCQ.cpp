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

#include "spectrumCQ.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* SpectrumCQ::name = "SpectrumCQ";
const char* SpectrumCQ::category = "Tonal";
const char* SpectrumCQ::description = DOC("This algorithm computes the Constant Q Spectrogram.\n"
"\n");


void SpectrumCQ::configure() {
  
  _sampleRate = parameter("sampleRate").toDouble();
  _minFrequency = parameter("minFrequency").toDouble();
  _maxFrequency = parameter("maxFrequency").toDouble();
  _binsPerOctave = parameter("binsPerOctave").toInt();
  _threshold = parameter("threshold").toDouble();

  // set temp port here as it's not gonna change between consecutive calls
  // to compute()
  _constantq->configure("minFrequency", _minFrequency,
                        "maxFrequency", _maxFrequency,
                        "binsPerOctave", _binsPerOctave,
                        "sampleRate", _sampleRate,
                        "threshold", _threshold);
  
  _constantq->output("constantq").set(_CQBuffer);
  _magnitude->input("complex").set(_CQBuffer);
}

void SpectrumCQ::compute() {

  const vector<Real>& signal = _signal.get();
  vector<Real>& spectrumCQ = _spectrumCQ.get();

  // Convert signal from Real to complex
  vector<complex<Real> > signalC(signal.begin(), signal.end());

  // Compute ConstantQ
  _constantq->input("frame").set(signalC);
  _constantq->compute();
  
  // Compute magnitude spectrum
  _magnitude->output("magnitude").set(spectrumCQ);
  _magnitude->compute();
}
