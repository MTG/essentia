/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "harmonicpeaks.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* HarmonicPeaks::name = "HarmonicPeaks";
const char* HarmonicPeaks::description = DOC("This algorithm finds the harmonic peaks of a signal given its spectral peaks and its fundamental frequency.\n"
"Note:\n"
"  - Input pitch is given as a hint in order to consider the closest spectral peak as the fundamental frequency.\n"
"  - Frequency and magnitude vectors must be sorted in ascending order.\n"
"\n"
"This algorithm is intended to receive its \"frequencies\" and \"magnitudes\" inputs from the SpectralPeaks algorithm.\n"
"\n"
"When input vectors differ in size or are empty, an exception is thrown. Input vectors must be ordered by ascending frequency excluding DC components and not contain duplicates, otherwise an exception is thrown.\n"
"If \"pitch\" is zero, it means its value is unknown, or the sound is unpitched, and in that case the HarmonicPeaks algorithm returns an empty vector.\n"
"\n"
"References:\n"
"  [1] Harmonic Spectrum - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Harmonic_spectrum");

void HarmonicPeaks::compute() {

  const vector<Real>& frequencies = _frequencies.get();
  const vector<Real>& magnitudes = _magnitudes.get();
  const Real& pitch = _pitch.get();
  vector<Real>& harmonicFrequencies = _harmonicFrequencies.get();
  vector<Real>& harmonicMagnitudes = _harmonicMagnitudes.get();

  if (magnitudes.size() != frequencies.size()) {
    throw EssentiaException("HarmonicPeaks: frequency and magnitude input vectors must have the same size");
  }

  if (pitch < 0) {
    throw EssentiaException("HarmonicPeaks: input pitch must be greater than zero");
  }

  harmonicFrequencies.resize(0);
  harmonicMagnitudes.resize(0);

  if (pitch == 0) {
    // pitch is unknown -> no harmonic peaks found
    return;
  }

  if (frequencies.empty()) {
    // no peaks -> no harmonic peaks either
    return;
  }


  // looking for f0
  Real f0 = frequencies[0];
  if (f0 <= 0) {
    throw EssentiaException("HarmonicPeaks: spectral peak frequencies must be greater than 0Hz");
  }

  Real m0 = magnitudes[0];
  Real errorMin = abs(f0 - pitch);
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
    Real error = abs(frequencies[i] - pitch);
    if (error <= errorMin) {
      f0 = frequencies[i];
      m0 = magnitudes[i];
      errorMin = error;
    }
  }

  Real log_2 = log10(2.0);
  for (int i=0; i<int(frequencies.size()); ++i) {
    Real semitones = fabs(12.0*log10(frequencies[i]/f0)/log_2);
    int semitonesRounded = int(round(semitones));
    if (semitonesRounded%12 == 0 && abs(semitones-semitonesRounded) <= 0.5) {
      harmonicFrequencies.push_back(frequencies[i]);
      harmonicMagnitudes.push_back(magnitudes[i]);
    }
  }
}
