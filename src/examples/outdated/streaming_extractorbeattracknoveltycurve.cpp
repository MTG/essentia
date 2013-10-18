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

#include "streaming_extractorbeattracknoveltycurve.h"
#include <iostream>
#include <fstream> // to write ticks to output file
#include <deque>
#include "algorithmfactory.h"
#include "network.h"
#include "poolstorage.h"
#include "vectorinput.h"
#include "vectoroutput.h"
#include "essentiamath.h"
#include "bpmutil.h"
#include "tnt/tnt2vector.h"

// NB: this file contains fuctions requred for outdated beat tracker back from
// 2009. It provided very low accuracy of beat trackign and has been replaced
// with RhythmExtractor2013. The functions are temporarily left in the case they
// can be further reused.

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

int bpmTolerance = 3;
Real maxBpm = 560; // leave it high unless you are sure about it
Real minBpm = 30;

void normalizeToMax(vector<Real>& array) {
  Real maxValue = -1.0*std::numeric_limits<int>::max();
  for (int i=0; i<int(array.size()); i++) {
    if (fabs(array[i]) > maxValue) maxValue = fabs(array[i]);
  }
  for (int i=0; i<int(array.size()); i++) array[i]/=maxValue;
}


vector<Real> computeNoveltyCurve(Pool& pool, const Pool& options) {

  Real sampleRate = options.value<Real>("analysisSampleRate");
  int hopSize = int(options.value<Real>("lowlevel.hopSize"));
  Real frameRate = sampleRate/Real(hopSize);
  standard::Algorithm* noveltyCurve = standard::AlgorithmFactory::create("NoveltyCurve",
                                                                         "frameRate", frameRate,
                                                                         "normalize", false,
                                                                         "weightCurveType",
                                                                         "flat");
  vector<Real> novelty;
  noveltyCurve->input("frequencyBands").set(pool.value<vector<vector<Real> > >("lowlevel.frequency_bands"));
  noveltyCurve->output("novelty").set(novelty);
  noveltyCurve->compute();
  delete noveltyCurve;
  normalizeToMax(novelty);

  // smoothing and derivative of hfc
  standard::Algorithm* mAvg = standard::AlgorithmFactory::create("MovingAverage",
                                                                 "size", int(0.1*frameRate));
  vector<Real> smoothHfc;
  mAvg->input("signal").set(pool.value<vector<Real> >("lowlevel.hfc"));
  mAvg->output("signal").set(smoothHfc);
  mAvg->compute();
  delete mAvg;
  normalizeToMax(smoothHfc);
  smoothHfc = derivative(smoothHfc);

  // adding 10% of hfc > 0 to novelty curve was found to be good when the genre
  // has percussive onsets
  // Note: that smoothing hfc will not add any extra delay to the novelty curve
  // because the NoveltyCurve algorithm internally smoothes it by the same amount
  for (int i=0; i<int(smoothHfc.size()); i++) {
    if (smoothHfc[i] > 0) novelty[i] += 0.1*smoothHfc[i];
  }

  vector<Real> envNovelty = novelty;

  // median filter
  int length=int(60./maxBpm*frameRate); // size of the window is max bpm (560)
  int size = envNovelty.size();
  novelty.resize(envNovelty.size());
  for (int i=0; i<size; i++) {
    int start = max(0, i-length);
    int end = min(start+2*length, size);
    if (end == size) start = end-2*length;
    vector<Real> window(envNovelty.begin()+start, envNovelty.begin()+end);
    Real m = essentia::median(window);
    novelty[i] = envNovelty[i] - m;
    if (novelty[i] < 0) novelty[i] = 0;
  }
  return novelty;
}

void fixedTempoEstimation(const vector<Real>& novelty, Real sampleRate,
                          Real hopSize, vector<Real>& bpms, vector<Real>& amplitudes) {
  standard::Algorithm* fixedTempoAlgo =
    standard::AlgorithmFactory::create("NoveltyCurveFixedBpmEstimator",
                                       "sampleRate", sampleRate,
                                       "hopSize", hopSize,
                                       "minBpm", minBpm,
                                       "maxBpm", maxBpm,
                                       "tolerance", bpmTolerance);
  fixedTempoAlgo->input("novelty").set(novelty);
  fixedTempoAlgo->output("bpms").set(bpms);
  fixedTempoAlgo->output("amplitudes").set(amplitudes);
  fixedTempoAlgo->compute();
  delete fixedTempoAlgo;
}

void mergeBpms(vector<Real>& bpmPositions, vector<Real>& bpmAmplitudes, Real tolerance) {
  vector<Real>::iterator posIter = bpmPositions.begin();
  vector<Real>::iterator ampsIter = bpmAmplitudes.begin();
  vector<Real>::iterator it1, it2;
  for (;posIter!=bpmPositions.end(); ++posIter, ++ampsIter) {
   it1 = posIter; it2=ampsIter;
   ++it1; ++it2;
    while(it1 != bpmPositions.end()) {
      if (areEqual(*posIter, *it1, tolerance)) {
        Real pos1 = *posIter;
        Real pos2 = *it1;
        Real amp1 = *ampsIter;
        Real amp2 = *it2;
        *posIter = (pos1*amp1+pos2*amp2)/(amp1+amp2);
        //*ampsIter += *it2;
        it1 = bpmPositions.erase(it1);
        it2 = bpmAmplitudes.erase(it2);
      }
      else {
        ++it1; ++it2;
      }
    }
  }
  for (int i=0;i<(int)bpmPositions.size(); ++i) {
    bpmPositions[i] = round(bpmPositions[i]);
  }
}

void computeEnergyTracks(const vector<vector<Real> >& tempogram,
                         const vector<Real>& bpms,
                         vector<Real>& resultBpms,
                         vector<Real>& resultAmps, Real tol) {
  resultBpms = bpms;
  resultAmps.resize(bpms.size(), 0);
  Real totalEnergy=0;
  for (int i=0; i<(int)tempogram.size(); i++) {
    Real currentEnergy = energy(tempogram[i]);
    if (currentEnergy == 0) continue;
    totalEnergy += currentEnergy;
    for (int j=0; j < (int)bpms.size(); j++) {
      int start = int(max(Real(0), bpms[j]-tol));
      int end   = int(min(Real(tempogram[i].size()-1), bpms[j]+tol));
      Real value = 0;
      for (int k=start; k<=end; k++) {
        value+=tempogram[i][k]*tempogram[i][k];
      }
      if (totalEnergy != 0) {
        resultAmps[j] += value/currentEnergy;
      }
    }
  }
  // normalize by the energy of the total tempogram so it does not depend on
  // the length of the audio
  for (int i=0; i<(int)resultAmps.size(); i++) {
    resultAmps[i] /= totalEnergy;
  }
  sortpair<Real, Real, greater<Real> >(resultAmps, resultBpms);
}

bool computeTempogram(const vector<Real>& noveltyCurve, Pool& results,
                      Real frameRate, Real tempoFrameSize, Real tempoOverlap,
                      int zeroPadding, Real inferredBpm=0) {
  VectorInput<Real>* gen = new VectorInput<Real>(&noveltyCurve);
  bool constantTempo = false;
  if (inferredBpm!=0) constantTempo = true;
  Algorithm* bpmHist = AlgorithmFactory::create("BpmHistogram",
                                                "frameRate", frameRate,
                                                "frameSize", tempoFrameSize,
                                                "zeroPadding", zeroPadding,
                                                "overlap", tempoOverlap,
                                                "maxPeaks", 50,
                                                "windowType", "blackmanharris92",
                                                "minBpm", minBpm,
                                                "maxBpm", maxBpm,
                                                "tempoChange", 5, // 5 seconds
                                                "constantTempo", constantTempo,
                                                "bpm", inferredBpm,
                                                "weightByMagnitude", true);
  connect(*gen, bpmHist->input("novelty"));

  connect(bpmHist->output("bpm"), results, "bpm");

  connect(bpmHist->output("bpmCandidates"), results, "bpmCandidates");
  connect(bpmHist->output("bpmMagnitudes"), results, "bpmMagnitudes");
  connect(bpmHist->output("tempogram"), results, "tempogram");
  connect(bpmHist->output("frameBpms"), results, "frameBpms");

  connect(bpmHist->output("ticks"), results, "ticks");
  connect(bpmHist->output("ticksMagnitude"), results, "ticksMagnitude");
  connect(bpmHist->output("sinusoid"), results, "sinusoid");

  Network network(gen);
  network.run();
  Real bpm = results.value<Real>("bpm");
  return bpm != 0;
}

Real computeMeanBpm(const vector<Real>& ticks) {
  int nticks = ticks.size();
  std::vector<Real> dticks(nticks-1);

  for (int i=0; i<nticks-1; i++) dticks[i] = ticks[i+1] - ticks[i];

  const int nbins = 100;
  std::vector<int> dist(nbins);
  std::vector<Real> distx(nbins);

  hist(&dticks[0], nticks-1, &dist[0], &distx[0], nbins);

  int maxidx = max_element(dist.begin(), dist.end()) - dist.begin();
  Real period = distx[maxidx];
  return 60./period;
}

bool computeBeats(const vector<Real>& noveltyCurve, Pool& results, Real frameRate,
                  Real tempoFrameSize, int tempoOverlap, int zeroPadding, Real bpm) {
  // compute the tempogram until the bpm and ticks stabilize...
  int count = 0;
  Real tol= 5;
  vector<Real> novelty = noveltyCurve;
  vector<vector<Real> > tempogram;
  while (tol < 20) {
    bool ok = computeTempogram(novelty, results, frameRate, tempoFrameSize,
                               tempoOverlap, zeroPadding, bpm);
    if (!ok) return false; // no beats found
    Real meanBpm = computeMeanBpm(results.value<vector<Real> >("ticks"));
    Real bpm = results.value<Real>("bpm");
    if (count == 0) { // first time we keep the original bpms
      results.add("first_tempogram", results.value<vector<TNT::Array2D<Real> > >("tempogram")[0]);
    }

    if (areEqual(bpm, meanBpm, tol))
        return true; // ticks and bpm stabilized. so quit!

    novelty.clear();
    novelty = results.value<vector<Real> >("sinusoid");
    results.remove("bpm");
    results.remove("bpmCandidates");
    results.remove("bpmMagnitudes");
    results.remove("frameBpms");
    results.remove("ticks");
    results.remove("ticksMagnitude");
    results.remove("sinusoid");
    results.remove("tempogram");
    count++;
    //tol += int(count/5.);
    //cout << "pass: " << count << endl;
    if (count%5==0) tol++;
  }
  return false;
}

void filterBpms(vector<Real>& bestBpms, vector<Real>& amplitudes,
                const vector<Real>& candidates, Real ceiling) {
  // this function tries to filter out the following issues from bestBpms:
  // 1. get rid of bpms > ceiling, by searching bpm/2 in candidates
  for (int i=0; i<(int)bestBpms.size(); i++) {
    if (bestBpms[i] > ceiling) {
      Real refBpm = bestBpms[i]/2.0;
      while (refBpm > 240) refBpm /= 2.0;
      for (int j=0; j<(int)candidates.size(); j++) {
        if (areEqual(refBpm, candidates[j], bpmTolerance)){
          bestBpms[i] = candidates[j];
          break;
        }
      }
    }
  }
  for (int i=0; i<(int)bestBpms.size(); i++) {
    for (int j=i+1; j<(int)bestBpms.size(); j++) {
      if (areEqual(bestBpms[i], bestBpms[j], bpmTolerance)) {
        bestBpms.erase(bestBpms.begin()+j);
        amplitudes.erase(amplitudes.begin()+j);
      }
    }
  }
}

void BeatTrack(Pool& pool, const Pool& options, const string& nspace) {

  vector<Real> novelty = computeNoveltyCurve(pool, options);

  Real sampleRate = options.value<Real>("analysisSampleRate");
  int hopSize = int(options.value<Real>("lowlevel.hopSize"));
  Real frameRate = sampleRate/Real(hopSize);
  Real tempoFrameSize = 4;  // 4 seconds minimum
  int tempoOverlap = 16;
  int zeroPadding = 1; // note that it is a factor, not a length

  Real bestBpm = 0;
  vector<Real> corrBpms, corrAmps;
  fixedTempoEstimation(novelty, sampleRate, hopSize, corrBpms, corrAmps);
  Pool results;
  bool ok = computeBeats(novelty, results, frameRate, tempoFrameSize, tempoOverlap, zeroPadding, bestBpm);
  if (ok) {
    vector<Real> bpms = results.value<vector<Real> >("bpmCandidates");
    vector<Real> bpmAmplitudes = results.value<vector<Real> >("bpmMagnitudes");
    mergeBpms(bpms, bpmAmplitudes, bpmTolerance);
    const TNT::Array2D<Real>& matrix = results.value<vector<TNT::Array2D<Real> > >("first_tempogram")[0];
    vector<vector<Real> > tempogram = array2DToVecvec(matrix);

    sortpair<Real,Real, greater<Real> > (bpmAmplitudes, bpms);

    vector<Real> finalBpms, ticksMagnitude;
    computeEnergyTracks(tempogram, bpms, finalBpms, ticksMagnitude, 3);
    normalize(ticksMagnitude);

    bpms.insert(bpms.end(), corrBpms.begin(), corrBpms.end());
    filterBpms(finalBpms, ticksMagnitude, bpms, 240); // bpms above 240 will be /2 if a harmonic exists

    bestBpm = finalBpms[0];

    // induce bestBpm in order to obtain the correct ticks:
    ok = computeBeats(novelty, results, frameRate, tempoFrameSize, tempoOverlap, zeroPadding, bestBpm);
  }

  // set namespace:
  string btspace = "beattrack.";
  if (!nspace.empty()) btspace = nspace + ".beattrack.";

  Real bpm = results.value<Real>("bpm");
  pool.set(btspace + "bpm", bpm);

  vector<Real> bpmCandidates = results.value<vector<Real> >("bpmCandidates");
  pool.set(btspace + "bpmCandidates", bpmCandidates);

  vector<Real> bpmMagnitudes = results.value<vector<Real> >("bpmMagnitudes");
  pool.set(btspace + "bpmMagnitudes", bpmMagnitudes);

  vector<Real> ticks = results.value<vector<Real> >("ticks");
  pool.set(btspace + "ticks", ticks);

  vector<Real> ticksMagnitude = results.value<vector<Real> >("ticksMagnitude");
  pool.set(btspace + "ticksMagnitude", ticksMagnitude);
}

