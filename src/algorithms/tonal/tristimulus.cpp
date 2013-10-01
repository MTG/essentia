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

#include "tristimulus.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Tristimulus::name = "Tristimulus";
const char* Tristimulus::description = DOC("This algorithm calculates the tristimulus of a signal given its harmonic peaks. The tristimulus has been introduced as a timbre equivalent to the color attributes in the vision. The tristimulus is composed of three different types of energy ratio allowing for a fine-grained description of the first harmonic of the spectrum, which are perceptually more salient.\n\n"

"Tristimulus is intended to be fed by the output of the HarmonicPeaks algorithm. The algorithm throws an exception when the input frequencies are not in ascending order and/or if the input vectors are of different sizes.\n\n"

"References:\n"
"  [1] Tristimulus (audio) - Wikipedia, the free encyclopedia\n"
"  http://en.wikipedia.org/wiki/Tristimulus_(audio)\n\n"
"  [2] G. Peeters, \"A large set of audio features for sound description\n" 
"  (similarity and classification) in the CUIDADO project,\" CUIDADO I.S.T.\n" 
"  Project Report, 2004");

void Tristimulus::compute() {

  const vector<Real>& frequencies = _frequencies.get();
  const vector<Real>& magnitudes = _magnitudes.get();
  vector<Real>& tristimulus = _tristimulus.get();

  if (magnitudes.size() != frequencies.size()) {
    throw EssentiaException("Tristimulus: frequency and magnitude vectors are of different size");
  }

  for (int i=1; i<int(frequencies.size()); i++) {
    if (frequencies[i] <= frequencies[i-1]) {
       throw EssentiaException("Tristimulus: harmonic peaks are not ordered by frequency");
    }
  }

  tristimulus.resize(3);

  Real sum = 0.0;
  for (int i=0; i<int(magnitudes.size()); i++) {
    sum += magnitudes[i];
  }

  if (sum == 0.0) {
    tristimulus[0] = 0.0;
    tristimulus[1] = 0.0;
    tristimulus[2] = 0.0;
    return;
  }

  tristimulus[0] = magnitudes[0] / sum;

  if (frequencies.size() < 4) {
    tristimulus[1] = 0.0;
    tristimulus[2] = 0.0;
    return;
  }

  tristimulus[1] = (magnitudes[1] + magnitudes[2] + magnitudes[3]) / sum;

  if (frequencies.size() < 5) {
    tristimulus[2] = 0.0;
    return;
  }

  Real sum_4 = 0.0;
  for (int i=4; i<int(magnitudes.size()); i++) {
    sum_4 += magnitudes[i];
  }
  tristimulus[2] = sum_4 / sum;
}
