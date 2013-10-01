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

#include "derivativesfx.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* DerivativeSFX::name = "DerivativeSFX";
const char* DerivativeSFX::description = DOC("This algorithm returns two descriptors that are based on the derivative of a signal envelope.\n"
"\n"
"The first descriptor is calculated after the maximum value of the input signal occurred. It is the average of the signal's derivative weighted by its amplitude. This coefficient helps discriminating impulsive sounds, which have a steep release phase, from non-impulsive sounds. The smaller the value the more impulsive.\n"
"\n"
"The second descriptor is the maximum derivative, before the maximum value of the input signal occurred. This coefficient helps discriminating sounds that have a smooth attack phase, and therefore a smaller value than sounds with a fast attack.\n"
"\n"
"This algorithm is meant to be fed by the outputs of the Envelope algorithm. If used in streaming mode, RealAccumulator should be connected in between.\n"
"An exception is thrown if the input signal is empty.");

void DerivativeSFX::compute() {

  const vector<Real>& envelope = _envelope.get();
  Real& derAvAfterMax = _derAvAfterMax.get();
  Real& maxDerBeforeMax = _maxDerBeforeMax.get();

  if (envelope.empty()) {
    throw EssentiaException("DerivativeSFX: input signal is empty");
  }

  int max = argmax(envelope);

  // Derivative average, weighted by the amplitude, after the max amplitude
  Real num = 0.0;
  Real den = 0.0;
  Real tmp1 = 0.0;
  if (max > 0) {
    tmp1 = envelope[max-1];
  }
  for (int i=max; i<int(envelope.size()); ++i) {
    Real value = envelope[i];
    Real der = value - tmp1;

    num += der;
    den += value;

    tmp1 = value;
  }

  if (den == 0.0) {
    derAvAfterMax = 0.0;
  }
  else {
    derAvAfterMax = num / den;
  }

  // Maximum derivative before the max amplitude
  Real tmp2 = 0.0;
  maxDerBeforeMax = envelope[0];
  for (int i=0; i<=max; i++) {
    Real value = envelope[i];
    Real der = value - tmp2;

    if (der > maxDerBeforeMax) {
      maxDerBeforeMax = der;
    }

    tmp2 = value;
  }
}
