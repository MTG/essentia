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
const char* SpectrumCQ::description = DOC("This algorithm computes the magnitude of the Constant-Q spectrum. See ConstantQ algorithm for more details.\n");


void SpectrumCQ::configure() {

  _constantq->configure(INHERIT("minFrequency"), INHERIT("numberBins"),
                        INHERIT("binsPerOctave"), INHERIT("sampleRate"),
                        INHERIT("threshold"), INHERIT("scale"),
                        INHERIT("windowType"), INHERIT("minimumKernelSize"),
                        INHERIT("zeroPhase"));

  _constantq->output("constantq").set(_CQBuffer);
  _magnitude->input("complex").set(_CQBuffer);
}

void SpectrumCQ::compute() {

  const vector<Real>& signal = _signal.get();
  vector<Real>& spectrumCQ = _spectrumCQ.get();


  // Compute ConstantQ.
  _constantq->input("frame").set(signal);
  _constantq->compute();
  
  // Compute magnitude spectrum.
  _magnitude->output("magnitude").set(spectrumCQ);
  _magnitude->compute();
}
