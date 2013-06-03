/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "beatogram.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Beatogram::name = "Beatogram";
const char* Beatogram::version = "1.0";
const char* Beatogram::description = DOC("This algorithm filters the loudness matrix given by beatsloudness algorithm in order to keep only the most salient beat band representation.\n"
"This algorithm has been found to be useful for estimating time signatures.\n");

void Beatogram::configure() {
  _windowSize = parameter("size").toInt();
}

void Beatogram::compute() {
  const vector<Real>& loudness = _loudness.get();
  const vector<vector<Real> >& loudnessBand = _loudnessBandRatio.get();
  vector<vector<Real> >& beatogram = _beatogram.get();
  int nticks = loudnessBand.size();
  vector<Real> meanRatiosPerTick(nticks, 0.0);
  vector<Real> medianRatiosPerTick(nticks, 0.0);
  for (int i=0; i<nticks; i++) {
    meanRatiosPerTick[i] = mean(loudnessBand[i]);
    medianRatiosPerTick[i] = median(loudnessBand[i]);
  }
  // transpose loudnessBand so it is [bands x ticks]
  beatogram = transpose(loudnessBand);
  int nbands = beatogram.size();
  // compute mean ratios for each tick around a window of 16 beats:
  vector<vector<Real> > meanRatiosPerBand(nbands, vector<Real>(nticks));
  vector<vector<Real> > medianRatiosPerBand(nbands, vector<Real>(nticks));
  for (int iBand=0; iBand<nbands; iBand++) {
    for (int iTick=0; iTick<nticks; iTick++) {
      int start = max(0, iTick - _windowSize/2);
      int end = min(nticks, start + _windowSize); 
      if (end == nticks) start = end-_windowSize;
      const vector<Real>& band = beatogram[iBand];
      vector<Real> window(band.begin()+start, band.begin()+end);
      meanRatiosPerBand[iBand][iTick] = mean(window);
      medianRatiosPerBand[iBand][iTick] = median(window);
    }
  }
  // filter out beatogram:
  for (int iBand=0; iBand<nbands; iBand++) {
    for (int iTick=0; iTick<nticks; iTick++) {
      Real bandThresh = max(medianRatiosPerBand[iBand][iTick],
                            meanRatiosPerBand[iBand][iTick]);
      Real tickThresh = max(medianRatiosPerTick[iTick],
                            meanRatiosPerTick[iTick]);
      if (beatogram[iBand][iTick] < bandThresh &&
          beatogram[iBand][iTick] <= tickThresh) {
        beatogram[iBand][iTick] = 0.0;
      }
      else {
        beatogram[iBand][iTick] *= loudness[iTick];
      }
    }
  }
}
