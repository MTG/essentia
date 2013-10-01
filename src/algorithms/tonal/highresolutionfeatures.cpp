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

#include "highresolutionfeatures.h"
#include "essentiamath.h"
#include "peak.h"

using namespace std;
using namespace essentia::util; // peak class

namespace essentia {
namespace standard {

const char* HighResolutionFeatures::name = "HighResolutionFeatures";
const char* HighResolutionFeatures::description = DOC("This algorithm computes high-resolution chroma features from an HPCP vector. The vector's size must be a multiple of 12 and it is recommended that it be larger than 120. In otherwords, the HPCP's resolution should be 10 Cents or more.\n"
"The high-resolution features being computed are:\n"
"\n"
"  - Equal-temperament deviation: a measure of the deviation of HPCP local maxima with respect to equal-tempered bins. This is done by:\n"
"    a) Computing local maxima of HPCP vector\n"
"    b) Computing the deviations from equal-tempered (abs) bins and their average\n"
"\n"
"  - Non-tempered energy ratio: the ratio betwen the energy on non-tempered bins and the total energy, computed from the HPCP average\n"
"\n"
"  - Non-tempered peak energy ratio: the ratio betwen the energy on non tempered peaks and the total energy, computed from the HPCP average\n"
"\n"
"HighFrequencyFeatures is intended to be used in conjunction with HPCP algorithm. Any input vector which size is not a positive multiple of 12, will raise an exception.\n"
"\n"
"References:\n"
"  [1] E. Gómez and P. Herrera, \"Comparative Analysis of Music Recordings\n"
"  from Western and Non-Western traditions by Automatic Tonal Feature\n"
"  Extraction,\" Empirical Musicology Review, vol. 3, pp. 140–156, 2008.");


vector<Peak> detectPeaks(const vector<Real>& hpcp, int maxPeaks) {
  vector<Peak> peaks;

  // wrap the hpcp around the first and last bin
  int size = int(hpcp.size());
  vector<Real> hpcpw(size + 2);
  hpcpw[0] = hpcp[size-1];
  for (int i=0; i<size; ++i) {
    hpcpw[i+1] = hpcp[i];
  }
  hpcpw[size+1] = hpcp[0];

  // find all peaks
  for (int i=1; i<size+1; ++i) {
    if (hpcpw[i-1] <= hpcpw[i] && hpcpw[i] >= hpcpw[i+1]) {
      peaks.push_back(make_pair(Real(i-1), hpcpw[i]));
    }
  }

  // sort them by descending amplitude
  sort(peaks.begin(), peaks.end(), greater<Peak>());

  if (int(peaks.size()) > maxPeaks) peaks.resize(maxPeaks);

  return peaks;
}

void HighResolutionFeatures::compute() {
  const vector<Real>& hpcp = _hpcp.get();

  const int hpcpSize = int(hpcp.size());
  const int binsPerSemitone = hpcpSize / 12;

  if (hpcpSize % 12 != 0 || hpcpSize == 0) {
    throw EssentiaException("HighResolutionFeatures: Cannot compute high-resolution features of an hpcp vector which size is not a non-zero multiple of 12");
  }

  // 1.- Equal-temperament deviation: measure of the deviation of HPCP local
  //     maxima with respect to equal-tempered bins.
  // a) Compute local maxima of HPCP vector
  //
  // should 24 be a parameter? Does this mean we are interested in 2 peaks per
  // semitone? -eaylon
  //
  // This doesn't mean 2 peaks per semitone, it means 24 peaks over the entire
  // hpcp vector. If there is a very high-resolution hpcp vector given, then
  // potentially, only the peaks in the first semitone will be detected. This
  // is OK however, because the peaks are truncated _after_ they are sorted by
  // amplitude. This means that we get the 24 largest peaks, which are assumed
  // to be the most relevant for this algorithm. As to whether it should become
  // a parameter, yes. I'll add it on the next commit. -rtoscano
  vector<Peak> peaks = detectPeaks(hpcp, parameter("maxPeaks").toInt());

  const int peaksSize = int(peaks.size());

  // b) replace the bin index by its deviation from equal-tempered bins
  for (int i=0; i<peaksSize; ++i) {
    // this could be changed by:
    Real f = peaks[i].position/ binsPerSemitone;
    Real dev = f - int(f);
    if (dev > 0.5) dev -= 1.0;
    peaks[i].position = dev;
  }

  // weight deviations by their amplitude
  Real eqTempDeviation = 0.0;
  Real totalWeights = 0.0;
  for (int i=0; i<peaksSize; ++i) {
    eqTempDeviation += abs(peaks[i].position * peaks[i].magnitude);
    totalWeights += peaks[i].magnitude;
  }

  if (totalWeights != 0.0) eqTempDeviation /= totalWeights;

  _equalTemperedDeviation.get() = eqTempDeviation;

  // 2.- NonTempered energy ratio: ratio betwen the energy on
  //     non-tempered bins and the total energy, computed from the HPCP average
  Real temperedEnergy = 0.0;
  Real totalEnergy = 0.0;
  for (int i=0; i<hpcpSize; ++i) {
    totalEnergy += hpcp[i] * hpcp[i];
    if (i % binsPerSemitone == 0) {
      temperedEnergy += hpcp[i] * hpcp[i];
    }
  }

  if (totalEnergy > 0.0) {
    _nt2tEnergyRatio.get() = 1.0 - temperedEnergy / totalEnergy;
  }
  else {
    _nt2tEnergyRatio.get() = 0.0;
  }

  // 3.- NonTempered peak energy ratio: ratio betwen the energy on
  //     non tempered peaks and the total energy, computed from the HPCP average
  Real temperedPeaksEnergy = 0.0;
  Real totalPeaksEnergy = 0.0;
  for (int i=0; i<peaksSize; ++i) {
    totalPeaksEnergy += peaks[i].magnitude * peaks[i].magnitude;
    if (peaks[i].position == 0.0) {
      temperedPeaksEnergy += peaks[i].magnitude * peaks[i].magnitude;
    }
  }

  if (totalPeaksEnergy > 0.0) {
    _nt2tPeaksEnergyRatio.get() = 1.0 - temperedPeaksEnergy / totalPeaksEnergy;
  }
  else {
    _nt2tPeaksEnergyRatio.get() = 0.0;
  }
}

} // namespace standard
} // namespace essentia


#include "poolstorage.h"
#include "algorithmfactory.h"

namespace essentia {
namespace streaming {

const char* HighResolutionFeatures::name = standard::HighResolutionFeatures::name;
const char* HighResolutionFeatures::description = standard::HighResolutionFeatures::description;


HighResolutionFeatures::HighResolutionFeatures() : AlgorithmComposite() {

  declareInput(_pcp, 1, 0, "hpcp", "the pitch class profile from which to detect the chord");

  declareOutput(_equalTemperedDeviation, 0, "equalTemperedDeviation",
                "measure of the deviation of HPCP local maxima with respect to equal-tempered bins");
  declareOutput(_nt2tEnergyRatio, 0, "nonTemperedEnergyRatio",
                "ratio between the energy on non-tempered bins and the total energy");
  declareOutput(_nt2tPeaksEnergyRatio, 0, "nonTemperedPeaksEnergyRatio",
                "ratio between the energy on non-tempered peaks and the total energy");


  _highResAlgo = standard::AlgorithmFactory::create("HighResolutionFeatures");
  _poolStorage = new PoolStorage<vector<Real> >(&_pool, "internal.highres_hpcp");

  _pcp  >>  _poolStorage->input("data");
}

void HighResolutionFeatures::configure() {
  _highResAlgo->configure(INHERIT("maxPeaks"));
}

AlgorithmStatus HighResolutionFeatures::process() {
  if (!shouldStop()) return PASS;

  //cout << _pool.value<vector<vector<Real> > >("internal.highres_hpcp") << endl;
  const vector<Real>& hpcp = _pool.value<vector<vector<Real> > >("internal.highres_hpcp")[0];
  Real deviation;
  Real energyRatio;
  Real peaksEnergyRatio;

  _highResAlgo->input("hpcp").set(hpcp);
  _highResAlgo->output("equalTemperedDeviation").set(deviation);
  _highResAlgo->output("nonTemperedEnergyRatio").set(energyRatio);
  _highResAlgo->output("nonTemperedPeaksEnergyRatio").set(peaksEnergyRatio);
  _highResAlgo->compute();

  _equalTemperedDeviation.push(deviation);
  _nt2tEnergyRatio.push(energyRatio);
  _nt2tPeaksEnergyRatio.push(peaksEnergyRatio);

  return FINISHED;
}

void HighResolutionFeatures::reset() {
  AlgorithmComposite::reset();
  _highResAlgo->reset();
}

} // namespace streaming
} // namespace essentia
