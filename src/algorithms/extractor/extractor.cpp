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

#include "extractor.h"
#include "algorithmfactory.h"
#include "essentiamath.h"
#include "poolstorage.h"
#include "vectorinput.h"
#include "network.h"

using namespace std;
using namespace essentia;
using namespace essentia::standard;

using streaming::connect;
using streaming::VectorInput;

const char* Extractor::name = "Extractor";
const char* Extractor::description = DOC("This algorithm extracts all low level, mid level and high level features from an audio signal and stores them in a pool.");

void Extractor::configure() {
  _lowLevelFrameSize = parameter("lowLevelFrameSize").toInt();
  _lowLevelHopSize = parameter("lowLevelHopSize").toInt();
  _tonalFrameSize = parameter("tonalFrameSize").toInt();
  _tonalHopSize = parameter("tonalHopSize").toInt();
  _dynamicsFrameSize = parameter("dynamicsFrameSize").toInt();
  _dynamicsHopSize = parameter("dynamicsHopSize").toInt();
  _sampleRate = parameter("sampleRate").toReal();
  _ns = parameter("namespace").toString();
  _llspace =     "lowLevel.";
  _sfxspace =    "sfx.";
  _rhythmspace = "rhythm.";
  _tonalspace =  "tonal.";
  if (!_ns.empty()) {
    _llspace =     _ns + ".lowLevel.";
    _sfxspace =    _ns + ".sfx.";
    _rhythmspace = _ns + ".rhythm.";
    _tonalspace =  _ns + ".tonal.";
  }
  _lowLevel = parameter("lowLevel").toBool();
  _tuning = parameter("tuning").toBool();
  _dynamics = parameter("dynamics").toBool();
  _rhythm = parameter("rhythm").toBool();
  _midLevel = parameter("midLevel").toBool();
  _highLevel = parameter("highLevel").toBool();
  _relativeIoi = parameter("relativeIoi").toBool();
}

void Extractor::compute() {
  const vector<Real>& signal = _signal.get();
  Pool& pool = _pool.get();
  VectorInput<Real> * gen = new VectorInput<Real>(&signal);
  if (_lowLevel) connectLowLevel(gen, pool);
  if (_rhythm) connectRhythm(gen, pool);
  if (_tuning) connectTuning(gen, pool);
  if (_dynamics) connectDynamics(gen, pool);

  scheduler::Network network(gen);
  network.run();
  if (_rhythm) postProcessOnsetRate(gen, pool);
  //deleteNetwork(gen); // done automatically when network goes out of scope
  if (_midLevel) computeMidLevel(signal, pool);
  if (_highLevel) computeHighLevel(pool);
  if (_relativeIoi) computeRelativeIoi(pool);
}

void Extractor::connectLowLevel(VectorInput<Real>* gen, Pool& pool) {
  streaming::Algorithm * lowLevel =
    streaming::AlgorithmFactory::create("LowLevelSpectralExtractor",
                                        "frameSize", _lowLevelFrameSize,
                                        "hopSize", _lowLevelHopSize,
                                        "sampleRate", _sampleRate);
  streaming::Algorithm * lowLevelEqloud =
    streaming::AlgorithmFactory::create("LowLevelSpectralEqloudExtractor",
                                        "frameSize", _lowLevelFrameSize,
                                        "hopSize", _lowLevelHopSize,
                                        "sampleRate", _sampleRate);
  // low level spectral:
  connect(*gen, lowLevel->input("signal"));
  const char * sfxDescArray[] = {"inharmonicity", "oddtoevenharmonicenergyratio", "tristimulus"};
  vector<string> sfxDesc = arrayToVector<string>(sfxDescArray);
  streaming::Algorithm::OutputMap::const_iterator it = lowLevel->outputs().begin();
  for (; it != lowLevel->outputs().end(); ++it) {
    string output_name = it->first;
    string ns = _llspace; // namespace
    if (contains(sfxDesc, output_name)) {
      ns = _sfxspace;
    }
    connect(*it->second, pool, ns + output_name);
  }

  // low level spectral eqloud:
  connect(*gen, lowLevelEqloud->input("signal"));
  it = lowLevelEqloud->outputs().begin();
  for (; it != lowLevelEqloud->outputs().end(); ++it) {
    connect(*it->second, pool, _llspace + it->first);
  }
}

void Extractor::connectDynamics(VectorInput<Real>* gen, Pool& pool) {
  // low level dynamics:
  streaming::Algorithm * level =
    streaming::AlgorithmFactory::create("LevelExtractor",
                                        "frameSize", _dynamicsFrameSize,
                                        "hopSize", _dynamicsHopSize);
  connect(*gen, level->input("signal"));
  connect(level->output("loudness"), pool, _llspace + "loudness");
}

void Extractor::connectTuning(VectorInput<Real>* gen, Pool& pool) {
  // low level tonal:
  streaming::Algorithm * tuning =
    streaming::AlgorithmFactory::create("TuningFrequencyExtractor",
                                        "frameSize", _tonalFrameSize,
                                        "hopSize", _tonalHopSize);
  connect(*gen, tuning->input("signal"));
  connect(tuning->output("tuningFrequency"), pool, _tonalspace + "tuning_frequency");
}

void Extractor::connectRhythm(VectorInput<Real>* gen, Pool& pool) {
  streaming::Algorithm * onsetRate = streaming::AlgorithmFactory::create("OnsetRate");
  connect(*gen, onsetRate->input("signal"));
  connect(onsetRate->output("onsetTimes"), pool, _rhythmspace + "onset_times");
  connect(onsetRate->output("onsetRate"), streaming::NOWHERE );
  streaming::Algorithm * rhythm = streaming::AlgorithmFactory::create("RhythmDescriptors");
  connect(*gen, rhythm->input("signal"));
  streaming::Algorithm::OutputMap::const_iterator it = rhythm->outputs().begin();
  for (; it != rhythm->outputs().end(); ++it) {
    connect(*it->second, pool, _rhythmspace + it->first);
  }
}

void Extractor::computeMidLevel(const vector<Real>& signal, Pool& pool) {
  if (!_tuning) {
    throw EssentiaException("Extractor: Mid level features depend on the tuning frequency. The algorithm should be reconfigured with the tuning parameter set to true");
  }
  Real tuningFreq = pool.value<vector<Real> >(_tonalspace + "tuning_frequency").back();

  VectorInput<Real> * gen = new VectorInput<Real>(&signal);
  streaming::Algorithm* tonal =
    streaming::AlgorithmFactory::create("TonalExtractor",
                                        "frameSize", _tonalFrameSize,
                                        "hopSize", _tonalHopSize,
                                        "tuningFrequency", tuningFreq);

  connect(*gen, tonal->input("signal"));
  streaming::Algorithm::OutputMap::const_iterator it = tonal->outputs().begin();
  for (; it != tonal->outputs().end(); ++it)
    connect(*it->second, pool, _tonalspace + it->first);

  if (_rhythm) {
    vector<Real> ticks = pool.value<vector<Real> >(_rhythmspace + "beats_position");
    streaming::Algorithm* beatsLoudness =
      streaming::AlgorithmFactory::create("BeatsLoudness",
                                          "sampleRate", _sampleRate,
                                          "beats", ticks);
    connect(*gen, beatsLoudness->input("signal"));
    connect(beatsLoudness->output("loudness"), pool, _rhythmspace + "beats_loudness");
    connect(beatsLoudness->output("loudnessBass"), pool, _rhythmspace + "beats_loudness_bass");
  }

  scheduler::Network network(gen);
  network.run();
}

void Extractor::computeHighLevel(Pool& pool) {
  if (_lowLevel) {
    levelAverage(pool);
    sfxPitch(pool);
    // clean up some tonal stuff:
    Real tuningFreq = pool.value<vector<Real> >(_tonalspace + "tuning_frequency").back();
    pool.remove(_tonalspace + "tuning_frequency");
    pool.set(_tonalspace + "tuning_frequency", tuningFreq);
  }
  if (_midLevel) {
    tuningSystemFeatures(pool);
    // remove the highres hpcp which were only used to compute other features
    pool.remove(_tonalspace + "hpcp_highres");
  }
}

void Extractor::computeRelativeIoi(Pool& p) {
  if (!_rhythm) {
    throw EssentiaException("Extractor: relative ioi depends on the rhythm features. The algorithm should be reconfigured with the rhythm parameter set to true");
  }
  const vector<string>& desc = p.descriptorNames();
  if (!contains(desc, _rhythmspace + "onset_times")) {
    p.add(_rhythmspace + "relative_ioi_peaks", TNT::Array2D<Real>());
    p.add(_rhythmspace + "relative_ioi", TNT::Array2D<Real>());
    return;
  }
  const vector<Real>& onsets = p.value<vector<Real> >(_rhythmspace + "onset_times");
  Real bpm = p.value<Real>(_rhythmspace + "bpm");
  int interp = 32; // 32th note interval
  Real interval = (60.0/bpm)/Real(interp);
  int size = onsets.size();
  if (bpm < 0 || onsets.empty()) {
    p.add(_rhythmspace + "relative_ioi_peaks", TNT::Array2D<Real>());
    p.add(_rhythmspace + "relative_ioi", TNT::Array2D<Real>());
    return;
  }
  vector<Real> riois;
  riois.reserve(size-1 + size-2 + size-3 + size-4);
  for (int i=1; i<size; ++i) riois.push_back((onsets[i]-onsets[i-1])/interval);
  for (int i=2; i<size; ++i) riois.push_back((onsets[i]-onsets[i-2])/interval);
  for (int i=3; i<size; ++i) riois.push_back((onsets[i]-onsets[i-3])/interval);
  for (int i=4; i<size; ++i) riois.push_back((onsets[i]-onsets[i-4])/interval);

  vector<Real> ioiDist;
  bincount(riois, ioiDist);
  vector<pair<Real,Real> > fullIoiDist(ioiDist.size());
  Real sumIoi = accumulate(ioiDist.begin(), ioiDist.end(), 0.0);
  for (int i=0; i<(int)fullIoiDist.size(); ++i) {
    fullIoiDist[i]=make_pair(Real(i)/Real(interp), ioiDist[i]/sumIoi);
  }
  // truncate fullIoiDist
  if (int(fullIoiDist.size()) > 5*interp) fullIoiDist.resize(size);
  // convert to array2D as the pool doesn't accept pairs...rrgggg!
  TNT::Array2D<Real> fullIoi(fullIoiDist.size(), 2);
  for (int i=0; i<fullIoi.dim1(); ++i) {
    fullIoi[i][0]=fullIoiDist[i].first;
    fullIoi[i][1]=fullIoiDist[i].second;
  }
  size = ioiDist.size();
  Algorithm * peakDetection = AlgorithmFactory::create("PeakDetection",
                                                       "minPosition", 0., "maxPosition", size,
                                                       "maxPeaks", 5, "range", size-1,
                                                       "interpolate", true, "orderBy", "amplitude");
  vector<Real> pos, values;
  peakDetection->input("array").set(ioiDist);
  peakDetection->output("positions").set(pos);
  peakDetection->output("amplitudes").set(values);
  peakDetection->compute();
  delete peakDetection;

  //vector<pair<Real, Real> > ioi_peaks(pos.size());
  TNT::Array2D<Real> ioi_peaks(pos.size(), 2);
  for (int i=0; i<int(pos.size()); ++i) {
    // scale back to 1 beat
    pos[i] /= Real(interp);
    // ratio across whole distribution surface:
    values[i] /= sumIoi;
    //ioi_peaks[i] = make_pair(pos[i], values[i]);
    ioi_peaks[i][0] = pos[i];
    ioi_peaks[i][1] = values[i];
  }
  p.add(_rhythmspace + "relative_ioi_peaks", ioi_peaks);
  p.add(_rhythmspace + "relative_ioi", fullIoi);
}

void Extractor::postProcessOnsetRate(VectorInput<Real>* gen, Pool& pool) {
  int nOnsets = 0;
  try {
    nOnsets = pool.value<vector<Real> >(_rhythmspace + "onset_times").size();
  }
  catch (const EssentiaException& ) { // no onsets found
    pool.set(_rhythmspace + "onset_rate", 0.0);
    return;
  }

  int nSamples =  (int)gen->output("data").totalProduced();
  pool.set(_rhythmspace + "onset_rate", nOnsets/(Real)nSamples*_sampleRate);
}

Real Extractor::squeezeRange(Real& x, Real& x1, Real& x2) {
  return (0.5 + 0.5 * tanh(-1.0 + 2.0 * (x - x1) / (x2 - x1)));
}

void Extractor::levelAverage(Pool& pool) {
  vector<Real> levelArray = pool.value<vector<Real> >(_llspace + "loudness");
  pool.remove(_llspace + "loudness");
  // Maximum dynamic
  Real EPSILON = 10e-5;
  Real maxValue = levelArray[argmax(levelArray)];
  if (maxValue <= EPSILON) {
    maxValue = EPSILON;
  }
  // Normalization to the maximum
  Real THRESHOLD = 0.0001; // this corresponds to -80dB
  for (int i=0; i<(int)levelArray.size(); i++) {
    levelArray[i] /= maxValue;
    if (levelArray[i] <= THRESHOLD) {
      levelArray[i] = THRESHOLD;
    }
  }
  // Average Level
  Real levelAverage = pow2db(mean(levelArray));

  // Re-scaling and range-control
  // This yields in numbers between
  // 0 for signals with  large dynamic variace and thus low dynamic average
  // 1 for signal with little dynamic range and thus
  // a dynamic average close to the maximum
  Real x1 = -5.0;
  Real x2 = -2.0;
  Real levelAverageSqueezed = squeezeRange(levelAverage, x1, x2);
  pool.set(_llspace + "average_loudness", levelAverageSqueezed);
}

void Extractor::tuningSystemFeatures(Pool& pool) {
  vector<Real> hpcp_highres = meanFrames(pool.value<vector<vector<Real> > >(_tonalspace + "hpcp_highres"));
  normalize(hpcp_highres);

  // 1- diatonic strength
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* keyDetect = factory.create("Key", "profileType", "diatonic");

  string key, scale;
  Real strength, unused;
  keyDetect->input("pcp").set(hpcp_highres);
  keyDetect->output("key").set(key);
  keyDetect->output("scale").set(scale);
  keyDetect->output("strength").set(strength);
  keyDetect->output("firstToSecondRelativeStrength").set(unused);
  keyDetect->compute();

  pool.set(_tonalspace + "tuning_diatonic_strength", strength);

  // 2- high resolution features
  Algorithm* highres = factory.create("HighResolutionFeatures");

  Real eqTempDeviation, ntEnergy, ntPeaks;
  highres->input("hpcp").set(hpcp_highres);
  highres->output("equalTemperedDeviation").set(eqTempDeviation);
  highres->output("nonTemperedEnergyRatio").set(ntEnergy);
  highres->output("nonTemperedPeaksEnergyRatio").set(ntPeaks);
  highres->compute();

  pool.set(_tonalspace + "tuning_equal_tempered_deviation", eqTempDeviation);
  pool.set(_tonalspace + "tuning_nontempered_energy_ratio", ntEnergy);

  // 3- THPCP
  vector<Real> hpcp = meanFrames(pool.value<vector<vector<Real> > >(_tonalspace + "hpcp"));
  normalize(hpcp);
  int idxMax = argmax(hpcp);
  vector<Real> hpcp_bak = hpcp;
  for (int i=idxMax; i<(int)hpcp.size(); i++) {
    hpcp[i-idxMax] = hpcp_bak[i];
  }
  int offset = hpcp.size() - idxMax;
  for (int i=0; i<idxMax; i++) {
    hpcp[i+offset] = hpcp_bak[i];
  }

  pool.set(_tonalspace + "thpcp", hpcp);

  delete keyDetect;
  delete highres;
}

void Extractor::sfxPitch(Pool& pool) {
  vector<Real> pitch = pool.value<vector<Real> >(_llspace + "pitch");

  Algorithm* maxtt = AlgorithmFactory::create("MaxToTotal");
  Real maxToTotal;
  maxtt->input("envelope").set(pitch);
  maxtt->output("maxToTotal").set(maxToTotal);
  maxtt->compute();
  pool.set(_sfxspace + "pitch_max_to_total", maxToTotal);

  Algorithm* mintt = AlgorithmFactory::create("MinToTotal");
  Real minToTotal;
  mintt->input("envelope").set(pitch);
  mintt->output("minToTotal").set(minToTotal);
  mintt->compute();
  pool.set(_sfxspace + "pitch_min_to_total", minToTotal);

  Algorithm* pc = AlgorithmFactory::create("Centroid");
  Real centroid;
  pc->configure("range", (uint)pitch.size() - 1);
  pc->input("array").set(pitch);
  pc->output("centroid").set(centroid);
  pc->compute();
  pool.set(_sfxspace + "pitch_centroid", centroid);

  Algorithm* amt = AlgorithmFactory::create("AfterMaxToBeforeMaxEnergyRatio");
  Real ratio;
  amt->input("pitch").set(pitch);
  amt->output("afterMaxToBeforeMaxEnergyRatio").set(ratio);
  amt->compute();
  pool.set(_sfxspace + "pitch_after_max_to_before_max_energy_ratio", ratio);

  delete maxtt;
  delete mintt;
  delete pc;
  delete amt;
}
