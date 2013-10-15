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

#include <iostream>
#include <fstream> // to write ticks to output file
#include <deque>
#include <essentia/algorithmfactory.h>
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/streaming/algorithms/vectorinput.h>
#include <essentia/streaming/algorithms/vectoroutput.h>
#include <essentia/essentiamath.h>
#include <essentia/scheduler/network.h>
#include <essentia/utils/bpmutil.h>
#include <essentia/utils/tnt/tnt2vector.h>

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;


int bpmTolerance = 3;
Real maxBpm = 560; // leave it high unless you are sure about it
Real minBpm = 30;

//TODO: get rid of TNT::Array2D once you have vector<vector< > >
//available in the pool or when you switch to eigen

bool computeBeats(const vector<Real>& noveltyCurve, Pool& pool, Real frameRate,
                  Real tempoFrameSize, int tempoOverlap, Real bpm=0);
void mergeBpms(vector<Real>& bpmPositions, vector<Real>& bpmAmplitudes, Real tolerance);

void normalizeToMax(vector<Real>& array) {
  Real maxValue = -1.0*std::numeric_limits<int>::max();
  for (int i=0; i<int(array.size()); i++) {
    if (fabs(array[i]) > maxValue) maxValue = fabs(array[i]);
  }
  for (int i=0; i<int(array.size()); i++) array[i]/=maxValue;
}


vector<Real> computeNoveltyCurve(Pool& pool, const string& audioFilename,
                                 int frameSize, int hopSize,
                                 Real startTime=0., Real endTime=2000.) {

  Real sampleRate = pool.value<Real>("sampleRate");

  // first compute the frequency bands:
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  Algorithm* audio = factory.create("EasyLoader",
                                    "filename",   audioFilename,
                                    "downmix", "mix",
                                    "startTime",  startTime,
                                    "endTime",    endTime,
                                    "sampleRate", sampleRate);
  Algorithm* fc = factory.create("FrameCutter",
                                 "frameSize", frameSize,
                                 "hopSize", hopSize,
                                 "silentFrames", "noise",
                                 "startFromZero", false);
  Algorithm* w = factory.create("Windowing",
                                "zeroPhase", false,
                                "type", "blackmanharris92");
  Algorithm* spectrum = factory.create("Spectrum");
  Algorithm* hfc = factory.create("HFC");
  Algorithm* freqBands = factory.create("FrequencyBands",
                                        "sampleRate", sampleRate);

  connect(audio->output("audio"), fc->input("signal"));
  connect(fc->output("frame"), w->input("frame"));
  connect(w->output("frame"), spectrum->input("frame"));
  connect(spectrum->output("spectrum"), freqBands->input("spectrum"));
  connect(spectrum->output("spectrum"), hfc->input("spectrum"));
  connect(freqBands->output("bands"), pool, "frequencyBands");
  connect(hfc->output("hfc"), pool, "hfc");

  Network network(audio);
  network.run();
  pool.set("audioSize", audio->output("audio").totalProduced()); // store length for further use


  Real frameRate = sampleRate/Real(hopSize);
  standard::Algorithm* noveltyCurve = standard::AlgorithmFactory::create("NoveltyCurve",
                                                                         "frameRate", frameRate,
                                                                         "normalize", false,
                                                                         "weightCurveType",
                                                                         "flat");
  vector<Real> novelty;
  noveltyCurve->input("frequencyBands").set(pool.value<vector<vector<Real> > >("frequencyBands"));
  noveltyCurve->output("novelty").set(novelty);
  noveltyCurve->compute();
  delete noveltyCurve;
  pool.remove("frequencyBands");
  pool.set("original_noveltyCurve", novelty);
  normalizeToMax(novelty);

  // smoothing and derivative of hfc
  standard::Algorithm* mAvg = standard::AlgorithmFactory::create("MovingAverage",
                                                                 "size", int(0.1*frameRate));
  vector<Real> smoothHfc;
  mAvg->input("signal").set(pool.value<vector<Real> >("hfc"));
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

  // compute envelope of the novelty curve
  //standard::Algorithm* envelope = standard::AlgorithmFactory::create("Envelope",
  //                                                                   "attackTime", 0.001*frameRate,
  //                                                                   "releaseTime",0.001*frameRate);
  //vector<Real> envNovelty;
  //envelope->input("signal").set(novelty);
  //envelope->output("signal").set(envNovelty);
  //envelope->compute();
  //delete envelope;
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

bool computeTempogram(const vector<Real>& noveltyCurve, Pool& pool,
                      Real frameRate, Real tempoFrameSize, Real tempoOverlap,
                      int zeroPadding, Real inferredBpm=0) {
  VectorInput<Real>* gen = new VectorInput<Real>(&noveltyCurve); //&pool.value<vector<Real> >("noveltyCurve"));
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
  //vector<vector<vector<Real> > > tempogramStorage;
  connect(*gen, bpmHist->input("novelty"));

  connect(bpmHist->output("bpm"), pool, "bpm");

  connect(bpmHist->output("bpmCandidates"), pool, "bpmCandidates");
  connect(bpmHist->output("bpmMagnitudes"), pool, "bpmMagnitudes");
  connect(bpmHist->output("tempogram"), pool, "tempogram");//tempogramStorage);
  connect(bpmHist->output("frameBpms"), pool, "frameBpms");

  connect(bpmHist->output("ticks"), pool, "ticks");
  connect(bpmHist->output("ticksMagnitude"), pool, "ticksMagnitude");
  connect(bpmHist->output("sinusoid"), pool, "sinusoid");

  Network network(gen);
  network.run();

  Real bpm = pool.value<Real>("bpm");
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

bool computeBeats(const vector<Real>& noveltyCurve, Pool& pool, Real frameRate,
                  Real tempoFrameSize, int tempoOverlap, int zeroPadding, Real bpm) {
  // compute the tempogram until the bpm and ticks stabilize...
  int count = 0;
  Real tol= 5;
  vector<Real> novelty = noveltyCurve;
  vector<vector<Real> > tempogram;
  while (tol < 20) {
    bool ok = computeTempogram(novelty, pool, frameRate, tempoFrameSize,
                               tempoOverlap, zeroPadding, bpm);
    if (!ok) return false; // no beats found
    Real meanBpm = computeMeanBpm(pool.value<vector<Real> >("ticks"));
    Real bpm = pool.value<Real>("bpm");
    if (count == 0) { // first time we keep the original bpms
      pool.add("first_tempogram", pool.value<vector<TNT::Array2D<Real> > >("tempogram")[0]);
    }
    if (areEqual(bpm, meanBpm, tol)) return true; // ticks and bpm stabilized. so quit!
    novelty.clear();
    novelty = pool.value<vector<Real> >("sinusoid");
    pool.remove("bpm");
    pool.remove("bpmCandidates");
    pool.remove("bpmMagnitudes");
    pool.remove("frameBpms");
    pool.remove("ticks");
    pool.remove("ticksMagnitude");
    pool.remove("sinusoid");
    pool.remove("tempogram");
    count++;
    //tol += int(count/5.);
    //cout << "pass: " << count << endl;
    if (count%5==0) tol++;
  }
  return false;
}

vector<Real> getAnnotations(const string& audioFilename) {
  string annotationFilename = audioFilename;
  string::size_type pos = annotationFilename.find("wav");
  while (pos != string::npos) {
    annotationFilename = annotationFilename.replace(pos, 3, "bpm");
    pos = annotationFilename.find("wav");
  }
  vector<Real> annotatedBpms;
  try {
    ifstream* fstream = new ifstream(annotationFilename.c_str());
    string line;
    while (getline(*fstream, line)) {
      annotatedBpms.push_back(atof(line.c_str()));
    }
    delete fstream;
  }
  catch(...) {
    cout << "annotation file \'" << annotationFilename << "\' not found"<< endl;
  }
  return annotatedBpms;
}


Real computeBeatsLoudness(const string& audioFilename, Pool& pool, Real sampleRate) {
  Real bands [] = { 0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0,
                    920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0,
                    3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0,
                    12000.0, 15500.0, 20500.0, 27000.0 };
  const vector<Real>& ticks = pool.value<vector<Real> >("ticks");
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  Algorithm* audio = factory.create("EasyLoader",
                                    "filename",   audioFilename,
                                    "downmix", "left",
                                    "startTime",  0,
                                    "endTime",    2000,
                                    "sampleRate", sampleRate);
  Algorithm* beatsLoudness = factory.create("BeatsLoudness",
                                            "sampleRate", sampleRate,
                                            "frequencyBands", arrayToVector<Real>(bands),
                                            "beats", ticks);
  connect(audio->output("audio"), beatsLoudness->input("signal"));
  connect(beatsLoudness->output("loudness"), pool, "loudness");
  connect(beatsLoudness->output("loudnessBandRatio"), pool, "loudnessBandRatio");
  Network network(audio);
  network.run();
  const vector<Real>& loudness = pool.value<vector<Real> >("loudness");
  const vector<vector<Real> >& loudnessRatio = pool.value<vector<vector<Real> > >("loudnessBandRatio");
  vector<Real> loudnessBand(ARRAY_SIZE(bands), 0);
  Real energy = 0;
  int n=0;
  for (int i=0; i<int(loudness.size()); i++) {
    if (loudness[i] > 1e-4) {
      energy += loudness[i];
      for (int j=0; j<(int)loudnessRatio[i].size(); j++) {
        loudnessBand[j] += loudnessRatio[i][j]*loudness[i];
      }
      n++;
    }
  }
  for (int i=0; i<(int)loudnessBand.size(); i++) {
    loudnessBand[i] /= Real(n);
  }
  //cout << "loundess Band per band: " << loudnessBand << endl;
  //return energy/Real(n);
  return loudnessBand[0]+loudnessBand[1]+loudnessBand[2];
}

void computeBeatogram(Pool& pool) {
  const vector<Real>& loudness = pool.value<vector<Real> >("loudness");
  vector<vector<Real> > loudnessBand = pool.value<vector<vector<Real> > >("loudnessBandRatio");

  standard::AlgorithmFactory& factory = standard::AlgorithmFactory::instance();

  standard::Algorithm* beatogramAlgo=factory.create("Beatogram", "size", 16);
  vector<vector<Real> > beatogram;
  beatogramAlgo->input("loudness").set(loudness);
  beatogramAlgo->input("loudnessBandRatio").set(loudnessBand);
  beatogramAlgo->output("beatogram").set(beatogram);
  beatogramAlgo->compute();
  delete beatogramAlgo;

  standard::Algorithm* meter=factory.create("Meter");
  Real timeSig;
  meter->input("beatogram").set(beatogram);
  meter->output("meter").set(timeSig);
  meter->compute();
  delete meter;
  cout << "Time signature: " << timeSig << endl;
}

vector<Real> getClosestMatch(const vector<Real>& bpms1, const vector<Real>& bpms2) {
  // finds the closest matches between bpms2 and bpm1 giving priority to bpms1
  // but keeping the values from bpms2.
  // Why? In principle bpms1 should be the bpms found by fixedTempoEstimator and
  // bpms2 would be the ones found by bpmHistogram. The reason to give priority
  // to bpms1, is due to the bpms from  fixedTempoEstimator tend to be lower,
  // however the bpms from bpmHistogram tend to be more exact.
  int n = 2;
  Real tolerance = 5; // be a bit more permissive at this point
  vector<Real> minDist(n, numeric_limits<int>::max());
  vector<Real> bestMatch(n,-1);
  vector<int> minIdx(n,bpms1.size());
  for (int i=0; i<(int)bpms1.size(); i++) {
    for (int j=0; j<(int)bpms2.size(); j++) {
      if (areEqual(bpms1[i],bpms2[j],tolerance)) {
        Real dist = i*i + j*j;
        Real meanBpm = round(bpms2[j]);
        if (dist < minDist[0] && i < minIdx[0] &&
            !areEqual(bestMatch[0],meanBpm,tolerance)) {
           bestMatch[1] = bestMatch[0];
           minDist[1]=minDist[0];
           minIdx[1]=minIdx[0];

           bestMatch[0] = meanBpm;
           minDist[0]=dist;
           minIdx[0]=i;
           break;
        }
        else if (dist < minDist[1] && i < minIdx[1] &&
                 !areEqual(bestMatch[0],meanBpm,tolerance) &&
                 !areEqual(bestMatch[1],meanBpm,tolerance)) {
           bestMatch[1] = meanBpm;
           minDist[1]=dist;
           minIdx[1]=i;
           break;
        }
      }
    }
  }
  vector<Real>::iterator iter = bestMatch.begin();
  while (iter != bestMatch.end() ) {
    if (*iter == -1)  iter = bestMatch.erase(iter);
    else ++iter;
  }
  return bestMatch;
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

void evaluateResults(const vector<Real>& bpms,
                     const vector<Real>& annotatedBpms) {
  // this function is only useful if you have a file with annotated bpms for a
  // specific song
  if (annotatedBpms.empty()) return;
  cout << "Evaluation: ";
  if (bpms.empty()) {
    cout << "FAIL\n";
    return;
  }
  vector<Real> error, ratio;
  for (int i=0; i<(int)bpms.size(); i++) {
    for (int j=0; j<(int)annotatedBpms.size(); j++) {
      Real e=0, r=0;
      bpmDistance(bpms[i], annotatedBpms[j], e, r);
      error.push_back(e); ratio.push_back(r);
    }
  }
  int octaveArray [] = {1,2,3,4,5,6,7,8,9,10,11,12};
  vector<int> octaves = arrayToVector<int>(octaveArray);

  // check whether we got the same octave and an error < 3:
  for (int j=0; j<(int)octaves.size(); j++) {
    for (int i=0; i<(int)ratio.size(); i++) {
      if (fabs(ratio[i]) == octaves[j] && fabs(error[i]) <= bpmTolerance) {
        cout << "OCTAVE " << int(ratio[i]) << endl;
        return;
      }
    }
  }
  cout << "ratio: " << ratio << endl;
  cout << "error: " << error << endl;
  cout << "FAIL\n";
}

void alignTicks(const string& audioFilename, Pool& pool, Real windowLength) {
  // As ticks are normally not so well aligned, this function tries to align
  // them with the nearest onset
  Real sampleRate = pool.value<Real>("sampleRate");
  standard::AlgorithmFactory& factory = standard::AlgorithmFactory::instance();
  standard::Algorithm* loader = factory.create("EasyLoader",
                                               "filename",   audioFilename,
                                               "downmix", "left",
                                               "startTime",  0,
                                               "endTime",    2000,
                                               "sampleRate", sampleRate);
  vector<Real> audio;
  loader->output("audio").set(audio);
  loader->compute();
  delete loader;
  Real audioLength = audio.size()/sampleRate;
  const vector<Real>& ticks = pool.value<vector<Real> >("ticks");
  int nticks = ticks.size();
  vector<Real> newTicks;
  newTicks.reserve(nticks);
  standard::Algorithm* trimmer = factory.create("Trimmer",
                                                "sampleRate", sampleRate);
  vector<Real> trimmedAudio;
  trimmer->input("signal").set(audio);
  trimmer->output("signal").set(trimmedAudio);
  int frameSize = 1024;
  int hopSize = frameSize/8;
  standard::Algorithm* fc = factory.create("FrameCutter",
                                           "startFromZero", true,
                                           "frameSize", frameSize,
                                           "hopSize", hopSize);
  standard::Algorithm* w = factory.create("Windowing");
  standard::Algorithm* spec = factory.create("Spectrum");
  standard::Algorithm* flux = factory.create("Flux");
  vector<Real> frame, windowedFrame, spectrum;
  Real fluxValue;
  fc->input("signal").set(trimmedAudio);
  fc->output("frame").set(frame);
  w->input("frame").set(frame);
  w->output("frame").set(windowedFrame);
  spec->input("frame").set(windowedFrame);
  spec->output("spectrum").set(spectrum);
  flux->input("spectrum").set(spectrum);
  flux->output("flux").set(fluxValue);
  for (int i=0; i<nticks; i++) {
    if (ticks[i] >= audioLength) break;
    Real startTime = max(ticks[i]-windowLength, Real(0));
    Real endTime = min(startTime+2*windowLength, audioLength);
    trimmer->configure("startTime", startTime, "endTime", endTime);
    trimmer->compute();
    vector<Real> fluxValues;
    fluxValues.reserve(trimmedAudio.size()/hopSize);
    while (true) {
      fc->compute();
      if (frame.empty()) break;
      w->compute();
      spec->compute();
      flux->compute();
      fluxValues.push_back(fluxValue);
    }
    fc->reset();
    int fluxSize = fluxValues.size();
    if (fluxSize<1) break;
    vector<Real> dfluxValues(fluxSize-1,0);
    for (int j=0; j<(int)fluxSize-1; j++) {
      Real delta = fluxValues[j+1]-fluxValues[j];
      if (delta>0) dfluxValues[j] = delta;
    }
    int maxIdx = argmax(dfluxValues);
    newTicks.push_back(startTime+Real(maxIdx*hopSize/sampleRate));
  }
  pool.remove("ticks");
  for (int i=0; i<(int)newTicks.size(); i++) pool.add("ticks", newTicks[i]);
  delete fc;
  delete w;
  delete spec;
  delete flux;
  delete trimmer;
}

int main(int argc, char* argv[]) {

  essentia::init();

  Pool pool;

  cout << "An outdated rhythm extractor (beat tracker, BPM) based on Novelty Curve (2009)." << endl;
  cout << "NOTE: this beat tracker is outdated (low accuracy compared to the new one), you might want to use streaming_rhythmextractor_multifeature instead." << endl;

  if (argc < 4 && argc != 3) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " audiofile output_ticks_file" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];

  vector<Real> annotatedBpms = getAnnotations(audioFilename);
  //if (annotatedBpms.empty()) return 0;

  cout << "**************************************************************\n";
  cout << "processing " << audioFilename <<endl;
  cout << "**************************************************************\n";
  cout << "Annotated bpm: " << annotatedBpms << endl;

  AlgorithmFactory& factory = AlgorithmFactory::instance();
  Algorithm* audio = factory.create("AudioLoader", "filename", audioFilename);
  Real sampleRate = audio->parameter("sampleRate").toReal();
  delete audio;

  pool.set("sampleRate", sampleRate);

  // parameters for the novelty curve:
  int frameSize = 1024;
  int hopSize = frameSize/2;

  // parameters for beat and tempo extraction:
  Real frameRate = sampleRate/Real(hopSize);
  Real tempoFrameSize = 4;  // 4 seconds minimum
  int tempoOverlap = 16;
  int zeroPadding = 1; // note that it is a factor, not a length

  Real startTime = 0;
  Real endTime = 2000;

  vector<Real> novelty = computeNoveltyCurve(pool, audioFilename,
                                             frameSize, hopSize,
                                             startTime, endTime);
  bool ok = false;
  Real bpm = 0;

  vector<Real> corrBpms, corrAmps;
  fixedTempoEstimation(novelty, sampleRate, hopSize, corrBpms, corrAmps);
  ok = computeBeats(novelty, pool, frameRate, tempoFrameSize, tempoOverlap, zeroPadding, bpm);
  if (ok) {
    vector<Real> bpms = pool.value<vector<Real> >("bpmCandidates");
    vector<Real> bpmAmplitudes = pool.value<vector<Real> >("bpmMagnitudes");
    mergeBpms(bpms, bpmAmplitudes, bpmTolerance);
    const TNT::Array2D<Real>& matrix = pool.value<vector<TNT::Array2D<Real> > >("first_tempogram")[0];
    vector<vector<Real> > tempogram = array2DToVecvec(matrix);
      //array2DToVecvec(pool.value<TNT::Array2D<Real> >("first_tempogram"));

    sortpair<Real,Real, greater<Real> > (bpmAmplitudes, bpms);
    //cout << "fft bpms: " <<bpms << "\t" << bpmAmplitudes << endl;
    //cout << "autocorrelation bpms: " << corrBpms << "\t" << corrAmps << endl;

    vector<Real> finalBpms, ticksMagnitude;
    computeEnergyTracks(tempogram, bpms, finalBpms, ticksMagnitude, 3);
    Real confidenceRef = ticksMagnitude[0];
    normalize(ticksMagnitude);

    bpms.insert(bpms.end(), corrBpms.begin(), corrBpms.end());
    filterBpms(finalBpms, ticksMagnitude, bpms, 240); // bpms above 240 will be /2 if a harmonic exists

    Real bestBpm = finalBpms[0];

    // induce bestBpm in order to obtain the correct ticks:
    ok = computeBeats(novelty, pool, frameRate, tempoFrameSize, tempoOverlap, zeroPadding, bestBpm);

    // finally merge the final candidates, in order to not have duplicates:
    //mergeBpms(finalBpms, ticksMagnitude, 3);
    cout << "bpms: " << finalBpms
         << "\tconfidence (ref. " << confidenceRef <<"): "
         << ticksMagnitude << endl;

    //evaluateResults(finalBpms, annotatedBpms);

    // write ticks to output file:
    //alignTicks(audioFilename, pool, 0.05);
    const vector<Real>& ticks = pool.value<vector<Real> > ("ticks");
    ostream* fileStream = new ofstream(outputFilename.c_str());
    for (int i=0; i<int(ticks.size()); i++) {
      *fileStream << ticks[i] << "\n";
    }
    delete fileStream;

    // just for testing. Create audio file with marked ticks:
    Algorithm* loader = factory.create("MonoLoader", "downmix", "left", "filename", audioFilename);
    string beatFilename = audioFilename.substr(0, audioFilename.rfind('.')) + "_beat.wav";

    Algorithm* onsetsMarker = factory.create("AudioOnsetsMarker",
                                             "onsets", pool.value<vector<Real> > ("ticks"));
    Algorithm* writer = factory.create("MonoWriter", "filename", beatFilename);

    connect(loader->output("audio"), onsetsMarker->input("signal"));
    connect(onsetsMarker->output("signal"), writer->input("audio"));

    Network network(loader);
    network.run();
  }
  else {
    cout << "No beats found or the bpm is too unstable.\n";
  }

  // this part is experimental and will try to guess the time signature of the
  // song from the loudness at each tick. To get acceptable results the ticks
  // should come from a high bpm. For instance, if we had candidate bpms: 50 and
  // 150, we should use the ticks from 150 as they give more information about
  // the rhythm than the ticks from 50
  computeBeatsLoudness(audioFilename, pool, sampleRate);
  computeBeatogram(pool);
  essentia::shutdown();

  return 0;
}
