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
const char* SpectrumCQ::description = DOC("This algorithm computes the Constant-Q spectrogram using FFT. See ConstantQ algorithm for more details.\n");


void SpectrumCQ::configure() {

  _constantq->configure(INHERIT("minFrequency"), INHERIT("numberBins"),
                        INHERIT("binsPerOctave"), INHERIT("sampleRate"),
                        INHERIT("threshold"));

  _fft->output("fft").set(_fftBuffer);
  _constantq->input("fft").set(_fftBuffer);
  _constantq->output("constantq").set(_CQBuffer);
  _magnitude->input("complex").set(_CQBuffer);
}

void SpectrumCQ::compute() {

  const vector<Real>& signal = _signal.get();
  vector<Real>& spectrumCQ = _spectrumCQ.get();

  // Compute FFT of the input signal.
  _fft->input("frame").set(signal);
  _fft->compute();

  // Compute ConstantQ.
  _constantq->compute();
  
  // Compute magnitude spectrum.
  _magnitude->output("magnitude").set(spectrumCQ);
  _magnitude->compute();
}
