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

#include "oddtoevenharmonicenergyratio.h"
#include <limits>

using namespace std;
using namespace essentia;
using namespace standard;

const char* OddToEvenHarmonicEnergyRatio::name = "OddToEvenHarmonicEnergyRatio";
const char* OddToEvenHarmonicEnergyRatio::description = DOC("This algorithm computes the ratio between a signal's odd and even harmonic energy given the signal's harmonic peaks. The odd to even harmonic energy ratio is a measure allowing to distinguish odd-harmonic-energy predominant sounds (such as from a clarinet) from equally important even-harmonic-energy sounds (such as from a trumpet). The required harmonic frequencies and magnitudes can be computed by the HarmonicPeals algorithm.\n"
"In the case when the even energy is zero, which may happen when only even harmonics where found or when only one peak was found, the algorithm outputs the maximum real number possible. Therefore, this algorithm should be used in conjunction with the harmonic peaks algorithm.\n"
"If no peaks are supplied, the algorithm outputs a value of one, assuming either the spectrum was flat or it was silent.\n"
"\n"
"An exception is thrown if the input frequency and magnitude vectors have different size. Finally, an exception is thrown if the frequency and magnitude vectors are not ordered by ascending frequency.\n"
"\n"
"References:\n"
"  [1] K. D. Martin and Y. E. Kim, \"Musical instrument identification:\n"
"  A pattern-recognition approach,\" The Journal of the Acoustical Society of\n"
"  America, vol. 104, no. 3, pp. 1768–1768, 1998.\n\n"
"  [2] K. Ringgenberg et al., \"Musical Instrument Recognition,\"\n"
"  http://cnx.org/content/col10313/1.3/pdf");

void OddToEvenHarmonicEnergyRatio::compute() {

  const vector<Real>& frequencies = _frequencies.get();
  const vector<Real>& magnitudes = _magnitudes.get();
  Real& oddtoevenharmonicenergyratio = _oddtoevenharmonicenergyratio.get();

  if (magnitudes.size() != frequencies.size()) {
    throw EssentiaException("OddToEvenHarmonicEnergyRatio: frequency and magnitude vectors have different size");
  }
  if (frequencies.empty()) {
    // if no peaks supplied then we assume the spectrum was flat or completely
    // silent, in which case it makes sense to throw a ratio = 1.0.
    oddtoevenharmonicenergyratio = 1.0;
    return;
  }

  Real even_energy = 0.0;
  Real odd_energy = 0.0;
  Real prevFreq = frequencies[0];

  for (int i=0; i<int(frequencies.size()); i++) {
    if (frequencies[i] < prevFreq) {
      throw EssentiaException("OddToEvenHarmonicEnergyRatio: harmonic peaks are not ordered by ascending frequency");
    }
    prevFreq = frequencies[i];

    if (i%2 == 0) even_energy += magnitudes[i] * magnitudes[i];
    else           odd_energy += magnitudes[i] * magnitudes[i];
  }

  if (even_energy == 0.0) {
     oddtoevenharmonicenergyratio = numeric_limits<Real>::max();
  }
  else {
     oddtoevenharmonicenergyratio = odd_energy / even_energy;
  }
}
