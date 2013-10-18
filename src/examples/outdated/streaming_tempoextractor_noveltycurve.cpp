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
#include <essentia/algorithmfactory.h>
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/streaming/algorithms/vectorinput.h>
#include <essentia/streaming/algorithms/vectoroutput.h>
#include <essentia/essentiamath.h>

using namespace std;
using namespace essentia;
using namespace essentia::streaming;


// helper functions which can also be found in bpmhistogram.cpp, but included
// here as to not include a cpp

template <typename T>
Real quantize(const T& val, int steps) {
  // quantizes the decimal part of val into 2.0*steps
  Real q=1.0/Real(2.0*steps);
  Real ival=Real(int(val));
  Real frac=val-ival;
  for (int i=0; i<2*steps; i++) {
    if ((frac >= i*q) && (frac<(i+1)*q)) {
      if (fabs(frac-i*q) < abs(frac-(i+1)*q)) return ival+i*q;
      return ival+(i+1)*q;
    }
  }
  throw EssentiaException("quantize should not have reached this point");
}

template <typename T, typename U>
bool areEqual(const T& a, const U& b, Real epsilon) {
  if (a<b) return areEqual(b, a, epsilon);
  return ((a-epsilon)<b) && ((a+epsilon)>b);
}

template<typename T>
bool areHarmonics(const T& x, const T& y, Real epsilon, bool bPower2=true) {
  if (y>x) return areHarmonics(y, x, epsilon, bPower2);
  Real a = quantize(x, 2);//quantizeReal(x); //Real(int(x+0.5));
  Real b = quantize(y, 2);//quantizeReal(y); //Real(int(y+0.5));
  if (a==T(0.0) || b==T(0.0)) {
    return false;
  }
  if (areEqual(a,b, epsilon)){
    return true;
  }
  Real ratio = quantize(a/b, 2);
  // TODO:we should compute the following instead:
  // because 310-150 should give a difference of 5bpm and not 10. As  310/2-150=5
  Real remainder = fabs(a/ratio-b);
  if (remainder<=epsilon) {
    if (ratio - int(ratio)) {
      return false;
    }
    if (bPower2) {
      return isPowerTwo(int(ratio)); // force harmonics to be power of two
    }
    return true; // accept any harmonic series: 1,2,3,4,5...
  }
  return false;
}


static const Real scheirerBands[] = { 0.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 22050.0 };
static const Real barkBands[] = { 0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0, 630.0,
                                  770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0,
                                  2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0,
                                  12000.0, 15500.0, 20500.0, 27000.0 };


Real roundToDecimal(Real x, int n) {
  // rounds a decimal number to the nth position
  Real k = pow(10.0, n);
  return int(x*k+0.5)/k;
}

void computeNoveltyCurve(Pool& pool, const string& audioFilename, int frameSize, int hopSize,
                         const vector<Real>& bands, const string& windowType, const string& weightCurve,
                         Real startTime=0., Real endTime=2000.) {

  Real sampleRate = pool.value<Real>("sampleRate");

  // first compute the frequency bands:
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  Algorithm* audio = factory.create("EasyLoader",
                                    "filename",   audioFilename,
                                    "downmix", "left",
                                    "startTime",  startTime,
                                    "endTime",    endTime,
                                    "sampleRate", sampleRate);
  Algorithm* fc = factory.create("FrameCutter",
                                 "frameSize", frameSize,
                                 "hopSize", hopSize,
                                 "dropSilentFrames", false,
                                 "startFromZero", true,
                                 "lastFrameToEndOfFile", true);
  Algorithm* w = factory.create("Windowing",
                                "zeroPhase", false,
                                "type", windowType);
  Algorithm* spectrum = factory.create("Spectrum");
  Algorithm* freqBands = factory.create("FrequencyBands",
                                        "sampleRate", sampleRate,
                                        "frequencyBands", bands);

  connect(audio->output("audio"), fc->input("signal"));
  connect(fc->output("frame"), w->input("frame"));
  connect(w->output("frame"), spectrum->input("frame"));
  connect(spectrum->output("spectrum"), freqBands->input("spectrum"));
  connect(freqBands->output("bands"), pool, "frequencyBands");

  runGenerator(audio);
  pool.set("audioSize", audio->output("audio").totalProduced()); // store length for further use
  deleteNetwork(audio);


  Real frameRate = sampleRate/Real(hopSize);
  standard::Algorithm*
    noveltyCurve = standard::AlgorithmFactory::create("NoveltyCurve",
                                                      "frameRate", frameRate,
                                                      "weightCurve", weightCurve);
  vector<Real> novelty;
  noveltyCurve->input("frequencyBands").set(pool.value<vector<vector<Real> > >("frequencyBands"));
  noveltyCurve->output("novelty").set(novelty);
  noveltyCurve->compute();
  delete noveltyCurve;
  pool.remove("frequencyBands");
  // add novelty curve to the pool. We keep the original one in order to use it
  // when aligning the sinusoids by cross-correlation
  pool.set("original_noveltyCurve", novelty);
  pool.set("noveltyCurve", novelty);

  // just for testing;
  //cout << "novelty: " << endl;
  //for (int i=0; i<(int)novelty.size(); i++){
  //  cout << novelty[i] << endl;
  //}
}

bool computeTempogram(Pool& pool, Real frameRate, Real tempoFrameSize, Real tempoOverlap) {
  VectorInput<Real>* gen = new VectorInput<Real>(&pool.value<vector<Real> >("noveltyCurve"));
  Algorithm* bpmHist = AlgorithmFactory::create("BpmHistogram",
                                                "frameRate", frameRate,
                                                "frameSize", tempoFrameSize,
                                                "overlap", tempoOverlap,
                                                "maxPeaks", 10,   // from here default values being used
                                                "windowType", "hann",
                                                "minBpm", 30.0,
                                                "maxBpm", 500.0,
                                                "tolerance", 5.0,
                                                "normalize", false,
                                                "weightByMagnitude", true);
  vector<vector<vector<Real> > > sinusoids;
  VectorOutput<vector<vector<Real> > >* storage = new VectorOutput<vector<vector<Real> > >(&sinusoids);
  connect(*gen, bpmHist->input("novelty"));
  connect(bpmHist->output("bpm"), pool, "peaksBpm");
  connect(bpmHist->output("bpmMagnitude"), pool, "peaksMagnitude");
  connect(bpmHist->output("harmonicBpm"), pool, "harmonicBpm");
  connect(bpmHist->output("ticks"), pool, "ticks");
  connect(bpmHist->output("sinusoid"), pool, "sinusoid");
  connect(bpmHist->output("sinusoids"), *storage); //pool, "sinusoids");

  runGenerator(gen);
  deleteNetwork(gen);

  if (pool.value<vector<Real> >("peaksBpm") == vector<Real>(1, 0)) {
    return false;
  }

  for (int i=0; i<int(sinusoids[0].size()); i++) {
    pool.add("sinusoids", sinusoids[0][i]);
  }

  // just for testing;
  //const vector<Real> & sine = pool.value<vector<Real> >("sinusoid");
  //cout << "sine: " << endl;
  //for (int i=0; i < (int)sine.size(); i++) {
  //  cout << sine[i] << endl;
  //}
  return true;
}

void cleanPool(Pool& pool) {
  pool.remove("noveltyCurve");
  pool.set("noveltyCurve", pool.value<vector<Real> >("sinusoid"));
  pool.remove("peaksBpm");
  pool.remove("peaksMagnitude");
  pool.remove("harmonicBpm");
  pool.remove("ticks");
  pool.remove("sinusoid");
  pool.remove("sinusoids");
}

int main(int argc, char* argv[]) {

  essentia::init();

  Pool pool;

  cout "BPM extractor based on Novetly Curve (probably outdated)" << endl;

  if (argc < 4 && argc != 3) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " inputfile outputfile" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];

  //cout << "==========================================\n";
  cout << "Processing " << audioFilename << endl;
  //cout << "==========================================\n";

  AlgorithmFactory& factory = AlgorithmFactory::instance();
  Algorithm* audio = factory.create("AudioLoader", "filename", audioFilename);
  Real sampleRate = audio->parameter("sampleRate").toReal();
  delete audio;

  pool.set("sampleRate", sampleRate);

  // parameters for the novelty curve:
  int frameSize = 1024;
  int hopSize = frameSize/2;
  string windowType = "hann";
  string weightCurve = "inverse_quadratic";
  vector<Real> bands = arrayToVector<Real>(barkBands);

  // parameters for beat and tempo extraction:
  Real frameRate = sampleRate/Real(hopSize);
  int tempoFrameSize = 4.; // 4 seconds minimum
  int tempoOverlap = 2;

  Real startTime = 0;
  Real endTime = 2000;
  vector<Real> bpms;
  // keep all the possible candidates and decide at the end which one we keep.
  // This is done only for the mirex tempo contest, where we have to give  2
  // bpms and the algorithm may throw only one candidate from harmonicBpm
  // output
  vector<Real> allBpmCandidates, allBpmMagnitudes;

  computeNoveltyCurve(pool, audioFilename, frameSize, hopSize, bands, windowType,
                      weightCurve, startTime, endTime);
  bool ok = computeTempogram(pool, frameRate, tempoFrameSize, tempoOverlap);
  if (ok) {
    while (bpms != pool.value<vector<Real> >("harmonicBpm")) {
      bpms = pool.value<vector<Real> >("harmonicBpm");
      const vector<Real> peaksBpm = pool.value<vector<Real> >("peaksBpm");
      const vector<Real> peaksMag = pool.value<vector<Real> >("peaksMagnitude");
      allBpmCandidates.insert(allBpmCandidates.end(), peaksBpm.begin(), peaksBpm.end());
      allBpmMagnitudes.insert(allBpmMagnitudes.end(), peaksMag.begin(), peaksMag.end());
      cleanPool(pool);
      computeTempogram(pool, frameRate, tempoFrameSize, tempoOverlap);
    }
    // find lower and higher harmonics to harmonicBpm found in the whole process:
    Real maxBpmBound = 300.;
    Real minBpmBound = 40.;

    const int nbins = 100;
    std::vector<int> dist(nbins);
    std::vector<Real> distx(nbins);

    for (int i=0; i<int(allBpmCandidates.size());i++) allBpmCandidates[i] /= 2.;

    // MEGA HACK: only for MIREX-2010 tempo estimation

    // group similar bpms:
    vector<Real> groupedBpms, groupedMagnitudes;
    groupedBpms.push_back(allBpmCandidates[0]);
    groupedMagnitudes.push_back(allBpmMagnitudes[0]);
    size_t size = 0;
    while (groupedBpms.size() != size) {
      size = groupedBpms.size();
      for (int i=1; i<int(allBpmCandidates.size());i++) {
        bool found = false;
        if (allBpmCandidates[i] < 0) continue;
        for (int j=0; j<int(groupedBpms.size()); j++) {
          if (areEqual(allBpmCandidates[i], groupedBpms[j], 5)) {
            Real sum = (allBpmMagnitudes[i]+groupedMagnitudes[j]);
            groupedBpms[j] = (allBpmCandidates[i]*allBpmMagnitudes[i]+ groupedBpms[j]*groupedMagnitudes[j])/sum;
            groupedMagnitudes[j] = sum/2.;
            allBpmCandidates[i] = -1;
            found=true;
            break;
          }
        }
        if (!found) {
          groupedBpms.push_back(allBpmCandidates[i]);
          groupedMagnitudes.push_back(allBpmMagnitudes[i]);
        }
      }
    }
    size = 0;
    allBpmCandidates = groupedBpms;
    allBpmMagnitudes = groupedMagnitudes;
    groupedBpms.clear();
    groupedMagnitudes.clear();
    groupedBpms.push_back(allBpmCandidates[0]);
    groupedMagnitudes.push_back(allBpmMagnitudes[0]);
    while (groupedBpms.size() != size) {
      size = groupedBpms.size();
      for (int i=1; i<int(allBpmCandidates.size());i++) {
        bool found = false;
        if (allBpmCandidates[i] < 0) continue;
        for (int j=0; j<int(groupedBpms.size()); j++) {
          if (areEqual(allBpmCandidates[i], groupedBpms[j], 5)) {
            Real sum = (allBpmMagnitudes[i]+groupedMagnitudes[j]);
            groupedBpms[j] = (allBpmCandidates[i]*allBpmMagnitudes[i]+ groupedBpms[j]*groupedMagnitudes[j])/sum;
            groupedMagnitudes[j] = sum/2.;
            allBpmCandidates[i] = -1;
            found=true;
            break;
          }
        }
        if (!found) {
          groupedBpms.push_back(allBpmCandidates[i]);
          groupedMagnitudes.push_back(allBpmMagnitudes[i]);
        }
      }
    }

    // put bpms and magnitudes into a map and sort it by greater:
    map<Real, Real, greater<Real> > bpmMap;
    for (int i=0; i<int(groupedBpms.size()); i++) {
      if (groupedBpms[i] < minBpmBound || groupedBpms[i] > maxBpmBound) continue;
      bpmMap.insert(pair<Real, Real>(groupedMagnitudes[i], groupedBpms[i]));
    }
    //cout << "BpmMap: " << endl;
    //map<Real, Real>::const_iterator it = bpmMap.begin();
    //for (; it!=bpmMap.end(); ++it) {
    //  cout << "bpm: " << it->second << "\t value: " << it->first << endl;
    //}

    vector<Real> bpmResult(2, 0.0), bpmRatio(2, 0.0), bpmStrength(2, 0.0);
    // assume that there's only one bpm (this is a request from MIREX -> songs
    // will have constant bpm for the tempo contest)

    Real harmonicBpm = bpms[0];

    // take the maximum as a good bpm if it is a harmonic of the harmonicBpm
    map<Real, Real>::const_iterator it = bpmMap.begin();
    for (; it!=bpmMap.end(); ++it) {
      if (areHarmonics(it->second, harmonicBpm, 5, false)) {
        bpmResult[0] = it->second;
        bpmRatio[0] = bpmResult[0]/harmonicBpm;
        if (bpmRatio[0] < 1) bpmRatio[0] = int(1./bpmRatio[0] + 0.5);
        else bpmRatio[0] = int(bpmRatio[0] + 0.5);
        if (bpmRatio[0] < 4.15) {
          bpmStrength[0] = it->first;
          break; // otherwise we may pick a too fast or too slow
        }
      }
    }

    // get the next harmonic
    it = bpmMap.begin();
    for (; it!=bpmMap.end(); ++it) {
      if ((areHarmonics(it->second, harmonicBpm, 5, false) ||
           areHarmonics(it->second, bpmResult[0], 5, true)) &&
          !areEqual(it->second, bpmResult[0], 5)) {
        bpmResult[1] = it->second;
        if (areHarmonics(bpmResult[1], harmonicBpm, 5, false )) {
          bpmRatio[1] = bpmResult[1]/harmonicBpm;
        }
        else bpmRatio[1] = bpmResult[1]/bpmResult[0];

        if (bpmRatio[1] < 1) bpmRatio[1] = int(1./bpmRatio[1] + 0.5);
        else bpmRatio[1] = int(bpmRatio[1] + 0.5);
        if (bpmRatio[1] < 4.15){
          bpmStrength[1] = it->first;
          break;
        }
        //break;
      }
    }


    if (bpmResult[0] == 0) {
      if (bpmResult[1] <= harmonicBpm) bpmResult[0] = bpmResult[1]*2.0;
      else bpmResult[0] = bpmResult[0]/2.0;
      bpmRatio[0] = 2.0;
    }
    if (bpmResult[1] == 0) {
      if (bpmResult[0] <= harmonicBpm) bpmResult[1] = bpmResult[0]*2.0;
      else bpmResult[1] = bpmResult[0]/2.0;
      bpmRatio[1] = 2.0;
    }
    if (bpmResult[0] > maxBpmBound) {
      bpmResult[0] /= bpmRatio[0];
    }
    if (bpmResult[0] < minBpmBound) {
      bpmResult[0] *= bpmRatio[0];
    }
    if (bpmResult[1] > maxBpmBound) {
      bpmResult[1] /= bpmRatio[1];
    }
    if (bpmResult[1] < minBpmBound) {
      bpmResult[1] *= bpmRatio[1];
    }

    if (bpmResult[0] > bpmResult[1]) {
      Real tempBpm = bpmResult[1];
      Real tempProb = bpmStrength[1];
      bpmResult[1] = bpmResult[0];
      bpmStrength[1] = bpmStrength[0];
      bpmResult[0] = tempBpm;
      bpmStrength[0] = tempProb;
    }

    // normalize probabilities:
    Real sum = bpmStrength[0] + bpmStrength[1];
    bpmStrength[0] /= sum;
    bpmStrength[1] /= sum;
    if (bpmStrength[1] == 1 || bpmStrength[0] == 1) {
      // it means that we couldn't find a lower or upper harmonic, so the
      // Strength should be 0.5
      bpmStrength[0] = bpmStrength[1] = 0.5;
    }
    //sort(bpmResult.begin(), bpmResult.end());


    // usually the interval of perceives tempos will never differ much more than
    // 3 times
    if (bpmResult[1]/bpmResult[0] < 1.4) {
      if (0.5*bpmResult[0] < minBpmBound && 2*bpmResult[1] < maxBpmBound) {
        bpmResult[1] *= 2.0;
      }
      else if (0.5*bpmResult[0] > minBpmBound && 2*bpmResult[1] > maxBpmBound) {
        bpmResult[0] /= 2.0;
      }
      else bpmResult[0] /= 2.0;
    }
    else if (bpmResult[1]/bpmResult[0] > 3.1) {
      bpmResult[0] *= 2.0;
    }
    // write ticks to output file:
    ostream* fileStream = new std::ofstream(outputFilename.c_str());
    *fileStream << bpmResult[0] << "\t" << bpmResult[1] << "\t" << bpmStrength[0] << endl;
    //cout << bpmResult[0] << "\t" << bpmResult[1] << "\t" << bpmStrength[0] << endl;
    delete fileStream;
  }
  else {
    // couldn't find a bpm... silence track?
    ostream* fileStream = new std::ofstream(outputFilename.c_str());
    *fileStream << 0 << "\t" << 0 << "\t" << 0 << endl;
    //cout << 0 << "\t" << 0 << "\t" << 0 << endl;
    delete fileStream;
  }


  essentia::shutdown();

  return 0;
}
