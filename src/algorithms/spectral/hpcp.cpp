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

#include "hpcp.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* HPCP::name = "HPCP";
const char* HPCP::description = DOC("Computes a Harmonic Pitch Class Profile (HPCP), that is a k*12 dimensional vector which represents the intensities of the twelve (k==1) semitone pitch classes, or subdivisions of these (k>1). It does this from the spectral peaks of a signal.\n"
"Regarding frequency parameters, exceptions are thrown if \"minFrequency\", \"splitFrequency\" and \"maxFrequency\" are not separated by at least 200Hz from each other, requiring that \"maxFrequency\" be greater than \"splitFrequency\" and \"splitFrequency\" be greater than \"minFrequenc\"."
"Other exceptions are thrown if input vectors have different size, if parameter \"size\" is not a positive non-zero multiple of 12 or if \"windowSize\" is less than one hpcp bin (12/size).\n"
"References:\n"
"  [1] T. Fujishima, \"Realtime Chord Recognition of Musical Sound: A System\n"
"  Using Common Lisp Music,\" in International Computer Music Conference\n"
"  (ICMC'99), pp. 464-467, 1999.\n"
"  [2] E. Gómez, \"Tonal Description of Polyphonic Audio for Music Content\n"
"  Processing,\" INFORMS Journal on Computing, vol. 18, no. 3, pp. 294–304,\n"
"  2006.");


const Real HPCP::precision = 0.00001;

void HPCP::configure() {
  _size = parameter("size").toInt();

  if (_size % 12 != 0) {
    throw EssentiaException("HPCP: The size parameter is not a multiple of 12.");
  }

  _windowSize = parameter("windowSize").toReal();

  if (_windowSize * _size/12 < 1.0) {
    throw EssentiaException("HPCP: Your windowSize needs to span at least one hpcp bin (windowSize >= 12/size)");
  }

  _referenceFrequency = parameter("referenceFrequency").toReal();
  _nHarmonics = parameter("harmonics").toInt();

  _minFrequency = parameter("minFrequency").toReal();
  _maxFrequency = parameter("maxFrequency").toReal();

  if ((_maxFrequency - _minFrequency) < 200.0) {
    throw EssentiaException("HPCP: Minimum and maximum frequencies are too close");
  }

  _splitFrequency = parameter("splitFrequency").toReal();
  _bandPreset = parameter("bandPreset").toBool();

  if (_bandPreset) {
    if ((_splitFrequency - _minFrequency) < 200.0) {
      throw EssentiaException("HPCP: Low band frequency range too small");
    }
    if ((_maxFrequency - _splitFrequency) < 200.0) {
      throw EssentiaException("HPCP: High band frequency range too small");
    }
  }

  string weightType = toLower(parameter("weightType").toString());
  if      (weightType == "none") _weightType = NONE;
  else if (weightType == "cosine") _weightType = COSINE;
  else if (weightType == "squaredcosine") _weightType = SQUARED_COSINE;
  else throw EssentiaException("Invalid weight type for HPCP: ", weightType);

  _nonLinear = parameter("nonLinear").toBool();
  _maxShifted = parameter("maxShifted").toBool();
  _normalized = parameter("normalized").toBool();

  if (_nonLinear && !_normalized) {
    throw EssentiaException("HPCP: Cannot apply non-linear filter when HPCP vector is not normalized");
  }

  initHarmonicContributionTable();
}


// Builds a weighting table of harmonic contribution. Higher harmonics
// contribute less and the fundamental frequency has a full harmonic
// strength of 1.0.
void HPCP::initHarmonicContributionTable() {
  _harmonicPeaks.clear();

  // Populate _harmonicPeaks with the semitonal positions of each of the
  // harmonics.
  for (int i = 0; i <= _nHarmonics; i++) {
    Real semitone = 12.0 * log2(i+1.0);
    Real octweight = max(1.0 , ( semitone /12.0)*0.5);

    // Get the semitone within the range (0-precision, 12.0-precision]
    while (semitone >= 12.0-precision) {
      semitone -= 12.0;
    }

    // Check to see if the semitone has already been added to _harmonicPeaks
    vector<HarmonicPeak>::iterator it;
    for (it = _harmonicPeaks.begin(); it != _harmonicPeaks.end(); it++) {
      if ((*it).semitone > semitone-precision && (*it).semitone < semitone+precision) break;
    }

    if (it == _harmonicPeaks.end()) {
      // no harmonic peak found for this frequency; add it
      _harmonicPeaks.push_back(HarmonicPeak(semitone, (1.0 / octweight)));
    }
    else {
      // else, add the weight
      (*it).harmonicStrength += (1.0 / octweight);
    }
  }
}


void HPCP::addContributionWithWeight(Real freq, Real mag_lin, vector<Real>& hpcp, Real harmonicWeight) const {
  int pcpSize = hpcp.size();
  Real resolution = pcpSize / 12; // # of bins / semitone

  // convert frequency in Hz to frequency in pcpBin index.
  // note: this can be a negative value
  Real pcpBinF = log2(freq / _referenceFrequency) * (Real)pcpSize;

  // which bins are covered by the window centered at this frequency
  // note: this is not wrapped.
  int leftBin = (int)ceil(pcpBinF - resolution * _windowSize / 2.0);
  int rightBin = (int)floor(pcpBinF + resolution * _windowSize / 2.0);

  assert(rightBin-leftBin >= 0);

  // apply weight to all bins in the window
  for (int i=leftBin; i<=rightBin; i++) {

    Real distance = abs(pcpBinF - (Real)i)/resolution;
    Real normalizedDistance = distance/_windowSize;
    Real weight = 0.;

    if (_weightType == COSINE) {
      weight = cos(M_PI*normalizedDistance);
    }
    else if (_weightType == SQUARED_COSINE) {
      weight = cos(M_PI*normalizedDistance);
      weight *= weight;
    }

    // here we wrap to stay inside the hpcp array
    int iwrapped = i % pcpSize;
    if (iwrapped < 0) iwrapped += pcpSize;

    hpcp[iwrapped] += weight * (mag_lin*mag_lin) * harmonicWeight * harmonicWeight;
  }
}


void HPCP::addContributionWithoutWeight(Real freq, Real mag_lin, vector<Real>& hpcp, Real harmonicWeight) const {
  if (freq <= 0)
    return;

  // Original Fujishima algorithm, basically places the contribution in the
  // bin nearest to the given frequency
  int pcpsize = hpcp.size();

  Real octave = log2(freq/_referenceFrequency);
  int pcpbin = (int)round(pcpsize * octave);  // bin distance from ref frequency

  pcpbin %= pcpsize;
  if (pcpbin < 0)
    pcpbin += pcpsize;

  hpcp[pcpbin] += mag_lin * mag_lin * harmonicWeight * harmonicWeight;
}


// Adds the magnitude contribution of the given frequency as the tonic
// semitone, as well as its possible contribution as a harmonic of another
// pitch.
void HPCP::addContribution(Real freq, Real mag_lin, vector<Real>& hpcp) const {
  vector<HarmonicPeak>::const_iterator it;

  for (it=_harmonicPeaks.begin(); it!= _harmonicPeaks.end(); it++) {
    // Calculate the frequency of the hypothesized fundmental frequency. The
    // _harmonicPeaks data structure always includes at least one element,
    // whose semitone value is 0, thus making this first iteration be freq == f
    Real f = freq * pow(2., -(*it).semitone / 12.0);
    Real harmonicWeight = (*it).harmonicStrength;

    if (_weightType != NONE) {
      addContributionWithWeight(f, mag_lin, hpcp, harmonicWeight);
    }
    else {
      addContributionWithoutWeight(f, mag_lin, hpcp, harmonicWeight);
    }
  }
}


void HPCP::compute() {
  const vector<Real>& frequencies = _frequencies.get();
  const vector<Real>& magnitudes = _magnitudes.get();
  vector<Real>& hpcp = _hpcp.get();

  // Check inputs
  if (magnitudes.size() != frequencies.size()) {
    throw EssentiaException("HPCP: Frequency and magnitude input vectors are not of equal size");
  }

  // Initialize data structures
  hpcp.resize(_size);
  fill(hpcp.begin(), hpcp.end(), (Real)0.0);

  vector<Real> hpcp_LO;
  vector<Real> hpcp_HI;

  if (_bandPreset) {
    hpcp_LO.resize(_size);
    fill(hpcp_LO.begin(), hpcp_LO.end(), (Real)0.0);

    hpcp_HI.resize(_size);
    fill(hpcp_HI.begin(), hpcp_HI.end(), (Real)0.0);
  }

  // Add each contribution of the spectral frequencies to the HPCP
  for (int i=0; i<(int)frequencies.size(); i++) {
    Real freq = frequencies[i];
    Real mag_lin = magnitudes[i];

    // Filter out frequencies not between min and max
    if (freq >= _minFrequency && freq <= _maxFrequency) {
      if (_bandPreset) {
        addContribution(freq, mag_lin, (freq < _splitFrequency) ? hpcp_LO : hpcp_HI);
      }
      else {
        addContribution(freq, mag_lin, hpcp);
      }
    }
  }

  // Normalize the HPCP vector
  if (_normalized) {
    if (_bandPreset) {
      normalize(hpcp_LO);
      normalize(hpcp_HI);
      for (int i=0; i<(int)hpcp.size(); i++) {
        hpcp[i] = hpcp_LO[i] + hpcp_HI[i];
      }
    }
    normalize(hpcp);
  } else {
    if (_bandPreset) {
      for (int i=0; i<(int)hpcp.size(); i++) {
        hpcp[i] = hpcp_LO[i] + hpcp_HI[i];
      }
    }
  }

  // Perform the Jordi non-linear post-processing step
  if (_nonLinear) {
    for (int i=0; i<(int)hpcp.size(); i++) {
      hpcp[i] = sin(hpcp[i] * M_PI * 0.5);
      hpcp[i] *= hpcp[i];
      if (hpcp[i] < 0.6) {
        hpcp[i] *= hpcp[i]/0.6 * hpcp[i]/0.6;
      }
    }
  }

  // Shift all of the elements so that the largest HPCP value is at index 0,
  // only if this option is enabled.
  if (_maxShifted) {
    int idxMax = argmax(hpcp);
    vector<Real> hpcp_bak = hpcp;
    for (int i=idxMax; i<(int)hpcp.size(); i++) {
      hpcp[i-idxMax] = hpcp_bak[i];
    }
    int offset = hpcp.size() - idxMax;
    for (int i=0; i<idxMax; i++) {
      hpcp[i+offset] = hpcp_bak[i];
    }
  }
}
