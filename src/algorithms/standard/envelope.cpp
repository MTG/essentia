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

#include "envelope.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Envelope::name = "Envelope";
const char* Envelope::description = DOC("This algorithm computes the envelope of a signal by applying a non-symmetric lowpass filter on a signal. By default it rectifies the signal, but that is optional.\n"
"\n"
"References:\n"
"  [1] U. Zölzer, Digital Audio Signal Processing,\n"
"  John Wiley & Sons Ltd, 1997, ch.7");

void Envelope::configure()
{
  Real samplerate = parameter("sampleRate").toReal();
  Real attackTime = parameter("attackTime").toReal() / 1000.f;
  Real releaseTime = parameter("releaseTime").toReal() / 1000.f;

  _ga = 0.0;
  if (attackTime > 0.0) {
    _ga = exp(- 1.0 / (samplerate * attackTime));
  }

  _gr = 0.0;
  if (releaseTime > 0.0) {
    _gr = exp(- 1.0 / (samplerate * releaseTime));
  }

  _applyRectification = parameter("applyRectification").toBool();

  reset();
}

void Envelope::reset() {
  _tmp = 0.0;
}

void Envelope::compute() {

  const vector<Real>& signal = _signal.get();
  vector<Real>& envelope = _envelope.get();

  envelope.resize(signal.size());

  for (int i=0; i<int(signal.size()); ++i) {

    Real sample = signal[i];
    if(_applyRectification) sample = fabs(sample);

    // we're in the attack phase
    if (_tmp < sample) {
      _tmp = (1.0 - _ga) * sample + _ga * _tmp;
    }

    // we're in the release phase
    else {
      _tmp = (1.0 - _gr) * sample + _gr * _tmp;
    }

    envelope[i] = _tmp;

    // prevent denormalization
    if (isDenormal(_tmp)) {
      _tmp = 0;
    }
  }
}
