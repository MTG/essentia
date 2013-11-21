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

#include "noveltycurvefixedbpmestimator.h"
#include "essentiamath.h"
#include "bpmutil.h"

using namespace std;
using namespace essentia;

namespace essentia {
namespace standard {

const char* NoveltyCurveFixedBpmEstimator::name = "NoveltyCurveFixedBpmEstimator";
const char* NoveltyCurveFixedBpmEstimator::version = "1.0";
const char* NoveltyCurveFixedBpmEstimator::description = DOC("Given the novelty curve (see NoveltyCurve algorithm), this algorithm outputs a histogram of the most probable bpms assuming the signal has constant tempo."
"This algorithm is based on the autocorrelation of the novelty curve and should only be used for signals that have a constant tempo or as a first tempo estimator to be used  in conjunction with other algorithms such as BpmHistogram."
"It is a simplified version of the algorithm described in [1] as, in order to predict the best BPM candidate,  it computes autocorrelation of the entire novelty curve instead of analyzing it on frames and histogramming the peaks over frames.\n"
"\n"
"References:\n"
"  [1] E. Aylon and N. Wack, \"Beat detection using plp,\" in Music Information\n"
"  Retrieval Evaluation Exchange (MIREX’10), 2010.\n");

void NoveltyCurveFixedBpmEstimator::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _hopSize = parameter("hopSize").toInt();
  _minBpm = parameter("minBpm").toReal();
  _maxBpm = parameter("maxBpm").toReal();
  _bpmTolerance = parameter("tolerance").toReal();
}

void NoveltyCurveFixedBpmEstimator::compute() {
  const vector<Real>& novelty = _novelty.get();
  vector<Real>& bpmPositions = _bpmPositions.get();
  vector<Real>& bpmAmplitudes = _bpmAmplitudes.get();


  vector<Real> corr;
  _autocor->input("array").set(novelty);
  _autocor->output("autoCorrelation").set(corr);
  _autocor->compute();

  Real minPos = int(bpmToLag(_maxBpm, _sampleRate, _hopSize)+0.5);
  vector<Real> maCorr;
  Algorithm* mavg=AlgorithmFactory::create("MovingAverage","size", minPos);
  mavg->input("signal").set(corr);
  mavg->output("signal").set(maCorr);
  mavg->compute();
  delete mavg;
  //cout << "correlation = " << corr << endl;

  // get all peaks from the correlation and compute a threshold based on the
  // most prominent ones
  int range = corr.size()-1;
  Algorithm* peakDetect=AlgorithmFactory::create("PeakDetection",
                                                 "range", range,
                                                 "orderBy", "position",
                                                 "interpolate", true,
                                                 "threshold", 0,
                                                 "maxPeaks", range,
                                                 "minPosition", minPos,
                                                 "maxPosition", range);
  vector<Real> peaksPositions, peaksAmplitudes;
  peakDetect->input("array").set(maCorr);
  peakDetect->output("positions").set( peaksPositions);
  peakDetect->output("amplitudes").set(peaksAmplitudes);
  peakDetect->compute();
  delete peakDetect;

  Real threshold = mainPeaksMean(peaksPositions, peaksAmplitudes, maCorr.size());
  while (true) {
    range = maCorr.size();
    Algorithm* peakDetect2=AlgorithmFactory::create("PeakDetection",
                                                    "range", range,
                                                    "orderBy", "position",
                                                    "interpolate", true,
                                                    "threshold", threshold,
                                                    "maxPeaks", range,
                                                    "minPosition", minPos,
                                                    "maxPosition", range);
    peakDetect2->input("array").set(maCorr);
    peakDetect2->output("positions").set( peaksPositions);
    peakDetect2->output("amplitudes").set(peaksAmplitudes);
    peakDetect2->compute();
    delete peakDetect2;

    int nPeaks = peaksPositions.size();
    vector<Real> peaksBpm;
    //int meters[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    //int meters[] = {1,2, 4, 8};
    int meters[] = {1};
    int nMeters = ARRAY_SIZE(meters);
    peaksBpm.reserve((nPeaks-1)*nMeters);
    for (int i=0; i<nPeaks; i++) {
      for (int j=0; j<nMeters; j++) {
        if (i+meters[j] > nPeaks-1) continue;
        Real diff = fabs(peaksPositions[i+meters[j]] - peaksPositions[i]);
        Real bpm = round(lagToBpm(diff, _sampleRate, _hopSize));
        if (bpm < _minBpm || bpm > _maxBpm) continue;
        peaksBpm.push_back(bpm);
      }
    }
    // construct a histogram + peak detection:
    //cout << "npeaks: " << peaksBpm.size() << endl;
    bpmPositions.clear();
    bpmAmplitudes.clear();
    histogramPeaks(peaksBpm, bpmPositions, bpmAmplitudes);
    inplaceMergeBpms(bpmPositions, bpmAmplitudes);
    sortpair<Real, Real, greater<Real> >(bpmAmplitudes, bpmPositions);
    //cout << "merged peaks: " << bpmPositions <<" " << bpmAmplitudes << endl;
    //if (!bpmPositions.empty()) {
    if (bpmPositions.size() > 2) {
      // need a minimum amount of peaks to ensure not skipping too much
      // information. TODO: are (at least) 3 peaks enough? seems so
      break;
    }
    threshold*=0.5;
  }
}

void NoveltyCurveFixedBpmEstimator::histogramPeaks(const vector<Real>& bpms,
                                       vector<Real>& positions,
                                       vector<Real>& amplitudes) {

  if (bpms.size() == 0) return;

  int nbins = 100;
  vector<int> dist(nbins);
  vector<Real> distx(nbins);
  int size = bpms.size();
  hist(&bpms[0], size, &dist[0], &distx[0], nbins);

  for (int i=0; i<4; i++) {
    // add a tail of zeros in case the maximum is right on the edge
    // and could be missed by the peak detector if we had bad luck
    // TODO: got some problems with peakdetection on a histogram when the peak is
    // at the last bin. Might be a bug in Peak Detection
    dist.push_back(0);
    distx.push_back(distx[nbins-1]+distx[nbins-1]-distx[nbins-2]);
    nbins++;
  }
  vector<Real> distReal(nbins);
  // convert to Real as peakdetection expects a vector<Real>
  for (int i=0; i<nbins; i++) distReal[i] = Real(dist[i]);
  int range = dist.size();
  Algorithm* peakDetect=AlgorithmFactory::create("PeakDetection",
                                                 "range", range,
                                                 "orderBy", "amplitude",
                                                 "interpolate", true,
                                                 "threshold", 0,
                                                 "maxPeaks", range,
                                                 "minPosition", 0,
                                                 "maxPosition", range);
  peakDetect->input("array").set(distReal);
  peakDetect->output("positions").set(positions);
  peakDetect->output("amplitudes").set(amplitudes);
  peakDetect->compute();
  delete peakDetect;
  for (int i=0; i<(int)positions.size(); i++) {
    positions[i] = round(distx[int(round(positions[i]))]);
  }
}

void NoveltyCurveFixedBpmEstimator::inplaceMergeBpms(vector<Real>& bpms,
                                         vector<Real>& amplitudes) {
  vector<Real>::iterator peaksIter = bpms.begin();
  vector<Real>::iterator ampsIter = amplitudes.begin();
  vector<Real>::iterator it1, it2;
  for (;peaksIter!=bpms.end(); ++peaksIter, ++ampsIter) {
   it1 = peaksIter; it2=ampsIter;
   ++it1; ++it2;
    while(it1 != bpms.end()) {
      if (areEqual(*peaksIter, *it1, _bpmTolerance)) {
        // should the peak be interpolated... for now yes
        Real bpm1 = *peaksIter;
        Real bpm2 = *it1;
        Real amp1 = *ampsIter;
        Real amp2 = *it2;
        *peaksIter = (bpm1*amp1+bpm2*amp2)/(amp1+amp2);
        *ampsIter += *it2;
        it1 = bpms.erase(it1);
        it2 = amplitudes.erase(it2);
      }
      else {
        ++it1;
        ++it2;
      }
    }
  }
}

Real NoveltyCurveFixedBpmEstimator::computeTatum(const vector<Real>& peaks) {
  // it is actually just a very rough estimation of the tatum
  int nPeaks = peaks.size();
  vector<Real> bpms; bpms.reserve(nPeaks-1);
  for (int i=1; i<nPeaks; i++) {
    Real diff = fabs(peaks[i] - peaks[i-1]);
    Real bpm = round(lagToBpm(diff, _sampleRate, _hopSize));
    if (bpm>_minBpm && bpm<_maxBpm) {
      bpms.push_back(bpm);
    }
  }
  vector<Real> peaksBpm, amplitudes;
  histogramPeaks(bpms, peaksBpm, amplitudes);
  sortpair<Real, Real, greater<Real> > (amplitudes, peaksBpm);
  return peaksBpm[0];
}

Real NoveltyCurveFixedBpmEstimator::mainPeaksMean(const vector<Real>& positions,
                                      const vector<Real>& amplitudes,
                                      int size) {
  // NOTE: peaks should be ordered by position
  int nPeaks = positions.size();
  // first get rid of very small peaks
  Real threshold = 0.1*min(median(amplitudes), mean(amplitudes));
  vector<Real> peaksPositions, peaksAmplitudes;
  peaksPositions.reserve(nPeaks);
  peaksAmplitudes.reserve(nPeaks);
  for (int i=0; i<nPeaks; i++) {
    if (amplitudes[i] < threshold) continue;
    peaksPositions.push_back(positions[i]);
    peaksAmplitudes.push_back(amplitudes[i]);
  }
  nPeaks = peaksPositions.size();
  Real tatumBpm = computeTatum(peaksPositions);
  int tatumLag = int(round(bpmToLag(tatumBpm, _sampleRate, _hopSize)));
  //cout << "tatum bpm: " << tatumBpm << endl;

  // find out which is the mean of the most relevant peaks.
  // slide in a window of about 16*tatum and keep the max amplitudes and
  // corresponding positions
  vector<Real> maxPositions;
  maxPositions.reserve(nPeaks);
  Real cumAmp = 0;
  int count = 0;
  int length = 4*tatumLag; //16*tatumLag;
  int lastIdx = -1;
  for (int i=0; i<nPeaks; i++) {
    int startpos = int(max(Real(0), Real(peaksPositions[i]-length)));
    int endpos = int(min(Real(size), Real(peaksPositions[i]+length+0.5)));
    // find out which peaks are closer to the corresponding start/end positions
    Real minDistFromStart = numeric_limits<int>::max();
    Real minDistFromEnd = numeric_limits<int>::max();
    int startIdx = numeric_limits<int>::max();
    int endIdx = numeric_limits<int>::max();
    for (int j=0; j<nPeaks; j++) {
      Real dist = fabs(peaksPositions[j] - startpos);
      if (dist<minDistFromStart) {
        minDistFromStart = dist;
        startIdx = j;
      }
      dist = fabs(peaksPositions[j] - endpos);
      if (dist<minDistFromEnd) {
        minDistFromEnd = dist;
        endIdx = j;
      }
    }
    // find maximum peak between startIdx and endIdx
    Real maxAmp = -1;
    int maxIdx = -1;
    for (int j=startIdx; j<=endIdx; j++) {
      if (maxAmp < peaksAmplitudes[j]) {
        maxAmp = peaksAmplitudes[j];
        maxIdx = j;
      }
    }
    if (maxAmp < 0 || maxIdx == lastIdx) continue;
    cumAmp += maxAmp;
    lastIdx = maxIdx;
    count++;
  }
  return cumAmp/Real(count);
}

} // namespace standard
} // namespace essentia
