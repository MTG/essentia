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

#include "dissonance.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Dissonance::name = "Dissonance";
const char* Dissonance::description = DOC("This algorithm calculates the sensory dissonance (to distinguish from musical or theoretical dissonance) of an audio signal given its spectral peaks. Sensory dissonance is based on the roughness of the spectral peaks."
"\n"
"Exceptions are thrown when the size of the input vectors are not equal or if input frequencies are not ordered ascendantly"
"\n"
"References:\n"
"  [1] R. Plomp and W. J. M. Levelt, \"Tonal Consonance and Critical\n"
"  Bandwidth,\" The Journal of the Acoustical Society of America, vol. 38,\n"
"  no. 4, pp. 548–560, 1965.\n\n"
"  [2] Critical Band - Handbook for Acoustic Ecology\n"
"  http://www.sfu.ca/sonic-studio/handbook/Critical_Band.html\n\n"
"  [3] Bark Scale -  Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Bark_scale");

Real aWeighting(Real f) {
  // from http://www.cross-spectrum.com/audio/weighting.html
  // 1.25893 = 2 dB
  return 1.25893*12200*12200*(f*f*f*f) / (
  (f*f +20.6*20.6) *
  (f*f +12200*12200) *
  sqrt(f*f +107.7*107.7) *
  sqrt(f*f +737.9*737.9)
  );
}

Real plompLevelt(Real df) {
  // df is the frequency difference on with critical bandwidth as  a unit.
  // the cooeficients were fitted with a polynom
  // to the data from the plomp & Levelt 1965 publication
  // To verify the fit run this and plot with e.g. gnuplot
  //
  //   #include <iostream>
  //   int main() {
  //       for (float i = 0; i <= 1.2; i+=0.01) {
  //           std::cout << plompLevelt(i) << std::endl;
  //       }
  //   }
  if (df < 0) return 1;
  if (df > 1.18) return 1;
  Real res =
      -6.58977878 * df*df*df*df*df +
      28.58224226 * df*df*df*df +
     -47.36739986 * df*df*df +
      35.70679761 * df*df +
     -10.36526344 * df +
       1.00026609;
  if (res < 0) return 0;
  if (res > 1) return 1;
  return res;
}

Real consonance(Real f1, Real f2) {
  // critical bandwidth between f1, f2:
  // see  http://www.sfu.ca/sonic-studio/handbook/Critical_Band.html for a
  // definition of critical bandwidth between two partials of a complex tone:
  Real cbwf1 = barkCriticalBandwidth(hz2bark(f1));
  Real cbwf2 = barkCriticalBandwidth(hz2bark(f2));
  Real cbw = std::min(cbwf1, cbwf2 );
  return plompLevelt(fabs(f2-f1)/cbw);
}



Real calcDissonance(const vector<Real>& frequencies, const vector<Real>& magnitudes) {
  vector<Real> loudness = magnitudes;
  Real totalLoudness = 0;
  int size = frequencies.size();

  // calculate dissonance
  for (int i = 0; i < size; i++) {
    // dBA-weighting
    // The factor should be applied to the amplitudes,
    // but we receive already the intensities (squared amplitudes),
    // thus, the factor is applied twice
    Real aWeightingFactor = aWeighting(frequencies[i]);
    loudness[i] *= aWeightingFactor * aWeightingFactor;
    totalLoudness += loudness[i];
  }


  if (totalLoudness == 0.0) {
    return 0.0;
  }

  //vector<Real> loudness(size);
  //for (int i=0; i<size; i++) partialLoudness = loudness[i]/totalLoudness;

  float totalDissonance = 0;
  for (int p1 = 0; p1 < size; p1++) {
    if (frequencies[p1] > 50) { // ignore frequencies below 50 Hz
      Real barkFreq = hz2bark(frequencies[p1]);
      Real startF = bark2hz(barkFreq - 1.18);
      Real endF = bark2hz(barkFreq + 1.18);
      int p2 = 0;
      Real peakDissonance = 0;
      while (p2 < size && frequencies[p2] < startF && frequencies[p2] < 50) p2++;
      while (p2 < size && frequencies[p2] < endF && frequencies[p2] < 10000) {
        float d = 1.0 - consonance(frequencies[p1], frequencies[p2]);
        // Dissonance from p1 to p2, should be the same as dissonance from p2
        // to p1, this is the reason for using both peaks' loudness as
        // weight
        if (d > 0) peakDissonance += d*(loudness[p2] + loudness[p1])/totalLoudness;
        p2++;
      }
      Real partialLoudness = loudness[p1]/totalLoudness;
      if (peakDissonance > partialLoudness) peakDissonance = partialLoudness;
      totalDissonance += peakDissonance;
    }
  }
  // total dissonance is divided by two, because each peak from a pair
  // contributes to it
  return totalDissonance/2;
}


void Dissonance::compute() {

  const vector<Real>& frequencies = _frequencies.get();
  const vector<Real>& magnitudes = _magnitudes.get();
  Real& dissonance = _dissonance.get();

  if (magnitudes.size() != frequencies.size()) {
    throw EssentiaException("Dissonance: frequency and magnitude input vectors are not the same size");
  }

  for (int i=1; i<int(frequencies.size()); i++) {
    if (frequencies[i] < frequencies[i-1]) {
      throw EssentiaException("Dissonance: spectral peaks must be sorted by frequency");
    }
  }

  dissonance = calcDissonance(frequencies, magnitudes);
}
