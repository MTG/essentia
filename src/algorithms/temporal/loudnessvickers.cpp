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

#include "loudnessvickers.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* LoudnessVickers::name = "LoudnessVickers";
const char* LoudnessVickers::description = DOC("This algorithm computes Vickers's loudness for a given audio signal. Currently, this algorithm only works for signals with a 44100Hz sampling rate. This algorithm is meant to be given frames of audio as input (not entire audio signals). The algorithm described in the paper performs a weighted average of the loudness value computed for each of the given frames, this step is left as a post processing step and is not performed by this algorithm.\n\n"

"References:\n"
"  [1] E. Vickers, \"Automatic Long-term Loudness and Dynamics Matching,\" in\n" 
"  The 111th AES Convention, 2001.");

void LoudnessVickers::configure() {

  // Vms initialization
  _Vms = 0.0;

  _sampleRate = parameter("sampleRate").toReal();

  vector<Real> b(2, 0.0);
  b[0] = 0.98595;
  b[1] = -0.98595;

  vector<Real> a(2, 0.0);
  a[0] = 1.0;
  a[1] = -0.9719;

  // Note: 0.035 is the time constant given in the paper
  _c = exp(-1.0 / (0.035 * _sampleRate));

  _filtering->configure("numerator", b, "denominator", a);
}

void LoudnessVickers::compute() {

  const vector<Real>& signal = _signal.get();
  Real& loudness = _loudness.get();

  // cheap B-curve loudness compensation
  vector<Real> signalFiltered;
  _filtering->input("signal").set(signal);
  _filtering->output("signal").set(signalFiltered);
  _filtering->compute();

  // create weight vector
  vector<Real> weight(signal.size(), 0.0);
  Real Vweight = 1.0;
  // create energy vector
  vector<Real> signalSquare(signal.size(), 0.0);

  for (int i=signal.size()-1; i>=0; --i) {
    weight[i] = Vweight;
    Vweight *= _c;
    signalSquare[i] = signalFiltered[i] * signalFiltered[i];
  }

  // update Vms
  _Vms = Vweight * _Vms + (1 - _c) * inner_product(weight.begin(), weight.end(), signalSquare.begin(), 0.0);

  // calculate loudness
  loudness = pow2db(_Vms); //10 * log10(_Vms + 1e-9);
}
