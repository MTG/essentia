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

#include "harmonicpeaks.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* HarmonicPeaks::name = "HarmonicPeaks";
const char* HarmonicPeaks::category = "Tonal";
const char* HarmonicPeaks::description = DOC("This algorithm finds the harmonic peaks of a signal given its spectral peaks and its fundamental frequency.\n"
"Note:\n"
"  - \"tolerance\" parameter defines the allowed fixed deviation from ideal harmonics, being a percentage over the F0. For example: if the F0 is 100Hz you may decide to allow a deviation of 20%, that is a fixed deviation of 20Hz; for the harmonic series it is: [180-220], [280-320], [380-420], etc.\n" 
"  - If \"pitch\" is zero, it means its value is unknown, or the sound is unpitched, and in that case the HarmonicPeaks algorithm returns an empty vector.\n"
"  - The output frequency and magnitude vectors are of size \"maxHarmonics\". If a particular harmonic was not found among spectral peaks, its ideal frequency value is output together with 0 magnitude.\n"
"This algorithm is intended to receive its \"frequencies\" and \"magnitudes\" inputs from the SpectralPeaks algorithm.\n"
"  - When input vectors differ in size or are empty, an exception is thrown. Input vectors must be ordered by ascending frequency excluding DC components and not contain duplicates, otherwise an exception is thrown.\n"
"\n"
"References:\n"
"  [1] Harmonic Spectrum - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Harmonic_spectrum");

bool sortCandidates(const std::pair<Real, std::pair<int, int> >& x, const std::pair<Real, std::pair<int, int> >& y) {
  return x.first < y.first;
}


void HarmonicPeaks::configure() {
  _maxHarmonics = parameter("maxHarmonics").toInt();
  _ratioTolerance = parameter("tolerance").toReal();
  _ratioMax = (Real) _maxHarmonics + _ratioTolerance;
}

void HarmonicPeaks::compute() {

  const vector<Real>& frequencies = _frequencies.get();
  const vector<Real>& magnitudes = _magnitudes.get();
  const Real& f0 = _pitch.get();
  vector<Real>& harmonicFrequencies = _harmonicFrequencies.get();
  vector<Real>& harmonicMagnitudes = _harmonicMagnitudes.get();

  if (magnitudes.size() != frequencies.size()) {
    throw EssentiaException("HarmonicPeaks: frequency and magnitude input vectors must have the same size");
  }

  if (f0 < 0) {
    throw EssentiaException("HarmonicPeaks: input pitch must be greater than zero");
  }

  harmonicFrequencies.resize(0);
  harmonicMagnitudes.resize(0);

  if (f0 == 0) {
    // pitch is unknown -> no harmonic peaks found
    return;
  }

  if (frequencies.empty()) {
    // no peaks -> no harmonic peaks either
    return;
  }

  if (frequencies[0] <= 0) {
    throw EssentiaException("HarmonicPeaks: spectral peak frequencies must be greater than 0Hz");
  }
  for (int i=1; i<int(frequencies.size()); ++i) {
    if (frequencies[i] < frequencies[i-1]) {
      throw EssentiaException("HarmonicPeaks: spectral peaks input must be ordered by frequency");
    }
    if (frequencies[i] == frequencies[i-1]) {
      throw EssentiaException("HarmonicPeaks: duplicate spectral peak found, peaks cannot be duplicated");
    }
    if (frequencies[i] <= 0) {
      throw EssentiaException("HarmonicPeaks: spectral peak frequencies must be greater than 0Hz");
    }
  }


  // Maximum allowed tolerance is less than 0.5 therefore, each peak can 
  // correspond only to one ideal harmonic

  // Init candidates with <-1, 0> -- ideal harmonics
  vector<pair<int, Real> > candidates (_maxHarmonics, make_pair(-1, 0));

  for (int i=0; i<int(frequencies.size()); ++i) {
    Real ratio = frequencies[i] / f0;
    int harmonicNumber = round(ratio);

    Real distance = abs(ratio - harmonicNumber);
    if (distance <= _ratioTolerance && ratio <= _ratioMax && harmonicNumber>0) {
      if (candidates[harmonicNumber-1].first == -1 || 
            distance < candidates[harmonicNumber-1].second) {
        // first occured candidate or a better candidate for harmonic
        candidates[harmonicNumber-1].first = i;
        candidates[harmonicNumber-1].second = distance;
      } 
      else if (distance == candidates[harmonicNumber-1].second) {
        // select the one with max amplitude
        if (magnitudes[i] > magnitudes[candidates[harmonicNumber-1].first]) {
          candidates[harmonicNumber-1].first = i;
          candidates[harmonicNumber-1].second = distance;
        }
      }
    }
  }

  for (int h=0; h < _maxHarmonics; ++h) {
    int i = candidates[h].first; 
    if (i < 0) {
      // harmonic not found, output ideal harmonic with 0 magnitude
      harmonicFrequencies.push_back((h+1) * f0);
      harmonicMagnitudes.push_back(0.);
    }
    else {
      harmonicFrequencies.push_back(frequencies[i]);
      harmonicMagnitudes.push_back(magnitudes[i]);
    }
  }
}
