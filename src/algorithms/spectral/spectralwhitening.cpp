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

#include "spectralwhitening.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* SpectralWhitening::name = "SpectralWhitening";
const char* SpectralWhitening::description = DOC("Performs spectral whitening of spectral peaks of a given spectrum. The algorithm works in dB scale, but the conversion is done by the algorithm so input should be in linear scale. The concept of 'whitening' refers to 'white noise' or a non-zero flat spectrum. It first computes a spectral envelope similar to the 'true envelope' in [1], and then modifies the amplitude of each peak relative to the envelope. For example, the predominant peaks will have a value close to 0dB because they are very close to the envelope. On the other hand, minor peaks between significant peaks will have lower amplitudes such as -30dB.\n"
"\n"
"The input \"frequencies\" and \"magnitudes\" can be computed using the SpectralPeaks algorithm.\n"
"\n"
"An exception is thrown if the input frequency and magnitude input vectors are of different size.\n"
"\n"
"References:\n"
"  [1] A. Röbel and X. Rodet, \"Efficient spectral envelope estimation and its\n"
"  application to pitch shifting and envelope preservation,\" in International\n"
"  Conference on Digital Audio Effects (DAFx’05), 2005.");

const Real SpectralWhitening::bpfResolution = 100.0;

void SpectralWhitening::configure() {
  _maxFreq = parameter("maxFrequency").toReal()*1.2;//1.2 magic number?
  _spectralRange = parameter("sampleRate").toReal() / 2.0;
}

void SpectralWhitening::compute() {
  const vector<Real>& spectrum = _spectrum.get();
  const vector<Real>& frequencies = _frequencies.get();
  const vector<Real>& magnitudes = _magnitudes.get();
  vector<Real>& magnitudesWhite = _magnitudesWhite.get();

  if (magnitudes.size() != frequencies.size()) {
    throw EssentiaException("SpectralWhitening: frequency and magnitude input vectors have different size");
  }

  const int nPeaks = magnitudes.size();
  vector<Real> magnitudesdB(nPeaks, 0.0);
  magnitudesWhite.resize(nPeaks);

  // If there are no magnitudes to whiten, do nothing
  if (nPeaks == 0) {
     return;
  }

  // Convert input linear magnitudes to dB scale
  for (int i=0; i<nPeaks; ++i) {
    magnitudesdB[i] = Real(2.0)*lin2db(magnitudes[i]);
  }

  // get max peak
  Real maxAmp = -numeric_limits<Real>::max();
  for (int i=0; i<nPeaks; ++i) {
    if (frequencies[i] <= _maxFreq) {
      maxAmp = max(maxAmp, magnitudesdB[i]);
    }
  }

  // compute envelope
  vector<Real> xPointsNoiseBPF;
  vector<Real> yPointsNoiseBPF;

  Real incr = bpfResolution;
  int specSize = spectrum.size();
  // reserve some meaningful space, i.e. size of sepctrum
  xPointsNoiseBPF.reserve(specSize);
  yPointsNoiseBPF.reserve(specSize);
  for (Real freq = 0.0; freq <= _maxFreq && freq <= _spectralRange; freq += incr) { //# magic numbers in the body of this for loop
    Real bf = freq - max(50.0, freq * 0.34); // 0.66
    Real ef = freq + max(50.0, freq * 0.58); // 1.58
    int b = int(bf / _spectralRange * (specSize - 1.0) + 0.5);
    int e = int(ef / _spectralRange * (specSize - 1.0) + 0.5);
    b = max(b, 0);
    b = min(specSize - 1, b);
    e = max(e, b + 1);
    e = min(specSize, e);
    Real c = b/2.0 + e/2.0;
    Real halfwindowlength = e - c;

    Real n = 0.0;
    Real wavg = 0.0;

    for (int i = b; i < e; ++i) {
      Real weight = 1.0 - abs(Real(i)-c) / halfwindowlength;
      weight *= weight;
      weight *= weight;
      Real spectrumEnergyVal = spectrum[i] * spectrum[i];
      weight *= spectrumEnergyVal;
      wavg += spectrumEnergyVal * weight;
      n += weight;
    }
    if (n != 0.0)
      wavg /= n;

    // Add points to the BPFs
    xPointsNoiseBPF.push_back(freq);
    yPointsNoiseBPF.push_back(wavg);
  }

  yPointsNoiseBPF[yPointsNoiseBPF.size() - 1] = yPointsNoiseBPF[yPointsNoiseBPF.size() - 2];

  for (int i=0; i<int(yPointsNoiseBPF.size()); ++i) {
    // don't optimise the sqrt as 0.5 outside lin2db as it fails for the case
    // 0, due to previously converted magnitudes to db
    yPointsNoiseBPF[i] = Real(2.0)*lin2db(sqrt(yPointsNoiseBPF[i]));
  }

  _noiseBPF.init(xPointsNoiseBPF, yPointsNoiseBPF);

  // compute envelope and peak difference to it
  for (int i=0; i<nPeaks; ++i) { //# lots of magic values below
    Real freq = frequencies[i];
    Real amp = magnitudesdB[i];

    if (freq > _maxFreq - incr) {
      // Copy over the magnitude to the output
      magnitudesWhite[i] = amp;

      continue; // This used to be a break-statement, but a break-statement
                // would only work if the "frequencies" and "magnitudesdB"
                // vectors were ordered by frequency
    }

    Real ampEnv = _noiseBPF(freq);
    if (amp < maxAmp - 40.0)
      magnitudesWhite[i] = (maxAmp - 40.0 - amp) / 2.0;
    if (amp > ampEnv)
      magnitudesWhite[i] = 0.0;
    else
      if (amp > ampEnv - 30.0)
        magnitudesWhite[i] = amp - ampEnv;
      else
        magnitudesWhite[i] = -200.0;
    magnitudesWhite[i] -= 20.0 * freq / 4000.0;
  }

  // Convert the whitened magnitudes back to linear scale
  for (int i=0; i<nPeaks; ++i) {
    // dividing by 2 due to converting to db => sqrt(lin2db(A)) lin2db(A/2)
    magnitudesWhite[i] = db2lin(magnitudesWhite[i]/2.0);
  }
}
