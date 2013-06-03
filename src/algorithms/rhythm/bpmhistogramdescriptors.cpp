/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "bpmhistogramdescriptors.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* BPMHistogramDescriptors::name = "BPMHistogramDescriptors";
const char* BPMHistogramDescriptors::description = DOC("This algorithm computes statistics for the highest and second highest peak of the beats per minute probability histogram.");

const int BPMHistogramDescriptors::maxBPM = 250; // max number of BPM bins
const int BPMHistogramDescriptors::numPeaks = 2;
const int BPMHistogramDescriptors::weightWidth = 3;
const int BPMHistogramDescriptors::spreadWidth = 9;

void BPMHistogramDescriptors::compute() {
  // copy input
  vector<Real> bpmValues = _bpmIntervals.get();

  // drop too-short intervals
  Real threshold = 60. / Real(maxBPM);
  vector<Real>::iterator i = bpmValues.begin();
  while (i != bpmValues.end()) {
    if (*i < threshold) {
      i = bpmValues.erase(i);
    }
    else {
      // convert values from interval to bpm
      bpmValues[i - bpmValues.begin()] = 60. / bpmValues[i - bpmValues.begin()];
      ++i;
    }
  }

  if (bpmValues.empty()) {
    _firstPeakBPM.get() = 0.0;
    _firstPeakWeight.get() = 0.0;
    _firstPeakSpread.get() = 0.0;

    _secondPeakBPM.get() = 0.0;
    _secondPeakWeight.get() = 0.0;
    _secondPeakSpread.get() = 0.0;

    return;
  }

  // compute histogram
  vector<Real> weights(maxBPM, 0.0);
  for (int i=0; i<int(bpmValues.size()); ++i) {
    int idx = min( maxBPM-1, int(round(bpmValues[i])));
    weights[idx]++;
  }

  // normalize histogram weights
  for (int i=0; i<int(weights.size()); ++i) {
    weights[i] /= bpmValues.size();
  }

  // peaks stats
  vector<Real> peakBPMs;
  vector<Real> peakWeights;
  vector<Real> peakSpreads;

  for (int i=0; i<numPeaks; ++i) {
    int idx = argmax(weights);

    Real peakBPM = idx;

    // peak weight is the weight of the peak and the weights of its two neighbors
    Real peakWeight
      = weights[idx]
      + (idx>0 ? weights[idx - ((weightWidth-1) / 2)] : 0)
      + (idx<int(weights.size())-1 ? weights[idx + ((weightWidth-1) / 2)] : 0);

    Real peakSpread = 0.0;
    int minIndex = max(idx - ((spreadWidth-1) / 2), 0);
    int maxIndex = min(idx + ((spreadWidth-1) / 2), int(weights.size()));

    for (int i=minIndex; i<=maxIndex; ++i) {
      peakSpread += weights[i];
      weights[i] = 0.0;
    }

    if (peakSpread > 0) {
      peakSpread = 1 - peakWeight / peakSpread;
    }

    peakBPMs.push_back(peakBPM);
    peakWeights.push_back(peakWeight);
    peakSpreads.push_back(peakSpread);
  }

  // output results
  _firstPeakBPM.get() = peakBPMs[0];
  _firstPeakWeight.get() = peakWeights[0];
  _firstPeakSpread.get() = peakSpreads[0];

  _secondPeakBPM.get() = peakBPMs[1];
  _secondPeakWeight.get() = peakWeights[1];
  _secondPeakSpread.get() = peakSpreads[1];
}
