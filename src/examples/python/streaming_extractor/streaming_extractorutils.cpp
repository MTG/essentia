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

#include "streaming_extractorutils.h"
#include "poolstorage.h"
#include "essentiamath.h" // for meanFrames and meanVariances

using namespace std;
using namespace essentia;

void readMetadata(const string& audioFilename, Pool& pool) {
  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();
  streaming::Algorithm* metadata = factory.create("MetadataReader",
                                                  "filename", audioFilename,
                                                  "failOnError", true);
  streaming::connect(metadata->output("title"),    pool, "metadata.tags.title");
  streaming::connect(metadata->output("artist"),   pool, "metadata.tags.artist");
  streaming::connect(metadata->output("album"),    pool, "metadata.tags.album");
  streaming::connect(metadata->output("comment"),  pool, "metadata.tags.comment");
  streaming::connect(metadata->output("genre"),    pool, "metadata.tags.genre");
  streaming::connect(metadata->output("track"),    pool, "metadata.tags.track");
  streaming::connect(metadata->output("year"),     pool, "metadata.tags.year");
  streaming::connect(metadata->output("length"),   streaming::NOWHERE); // let audio loader take care of this
  streaming::connect(metadata->output("bitrate"),  pool, "metadata.audio_properties.bitrate");
  streaming::connect(metadata->output("sampleRate"), streaming::NOWHERE); // let the audio loader take care of this
  streaming::connect(metadata->output("channels"), pool, "metadata.audio_properties.channels");
  runGenerator(metadata);
  deleteNetwork(metadata);
}

Real squeezeRange(Real& x, Real& x1, Real& x2) {
  return (0.5 + 0.5 * tanh(-1.0 + 2.0 * (x - x1) / (x2 - x1)));
}

void LevelAverage(Pool& pool, const string& nspace) {

  // namespace:
  string llspace = "lowlevel.";
  if (!nspace.empty()) llspace = nspace + ".lowlevel.";

  vector<Real> levelArray = pool.value<vector<Real> >(llspace + "loudness");
  pool.remove(llspace + "loudness");

  // Maximum dynamic
  Real EPSILON = 10e-5;
  Real maxValue = levelArray[argmax(levelArray)];
  if (maxValue <= EPSILON) {
    maxValue = EPSILON;
  }

  // Normalization to the maximum
  Real THRESHOLD = 0.0001; // this corresponds to -80dB
  for (uint i=0; i<levelArray.size(); i++) {
    levelArray[i] /= maxValue;
    if (levelArray[i] <= THRESHOLD) {
      levelArray[i] = THRESHOLD;
    }
  }

  // Average Level
  Real levelAverage = 10*log10(mean(levelArray));

  // Re-scaling and range-control
  // This yields in numbers between
  // 0 for signals with  large dynamic variace and thus low dynamic average
  // 1 for signal with little dynamic range and thus
  // a dynamic average close to the maximum
  Real x1 = -5.0;
  Real x2 = -2.0;
  Real levelAverageSqueezed = squeezeRange(levelAverage, x1, x2);
  pool.set(llspace + "average_loudness", levelAverageSqueezed);
}

void TuningSystemFeatures(Pool& pool, const string& nspace) {

  // namespace
  string tonalspace = "tonal.";
  if (!nspace.empty()) tonalspace = nspace + ".tonal.";

  vector<Real> hpcp_highres = meanFrames(pool.value<vector<vector<Real> > >(tonalspace + "hpcp_highres"));
  normalize(hpcp_highres);

  // 1- diatonic strength
  standard::AlgorithmFactory& factory = standard::AlgorithmFactory::instance();

  standard::Algorithm* keyDetect = factory.create("Key",
                                                  "profileType", "diatonic");

  string key, scale;
  Real strength, unused;
  keyDetect->input("pcp").set(hpcp_highres);
  keyDetect->output("key").set(key);
  keyDetect->output("scale").set(scale);
  keyDetect->output("strength").set(strength);
  keyDetect->output("firstToSecondRelativeStrength").set(unused);
  keyDetect->compute();

  pool.set(tonalspace + "tuning_diatonic_strength", strength);

  // 2- high resolution features
  standard::Algorithm* highres = factory.create("HighResolutionFeatures");

  Real eqTempDeviation, ntEnergy, ntPeaks;
  highres->input("hpcp").set(hpcp_highres);
  highres->output("equalTemperedDeviation").set(eqTempDeviation);
  highres->output("nonTemperedEnergyRatio").set(ntEnergy);
  highres->output("nonTemperedPeaksEnergyRatio").set(ntPeaks);
  highres->compute();

  pool.set(tonalspace + "tuning_equal_tempered_deviation", eqTempDeviation);
  pool.set(tonalspace + "tuning_nontempered_energy_ratio", ntEnergy);

  // 3- THPCP
  vector<Real> hpcp = meanFrames(pool.value<vector<vector<Real> > >(tonalspace + "hpcp"));
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

  pool.set(tonalspace + "thpcp", hpcp);

  delete keyDetect;
  delete highres;
}

void SFXPitch(Pool& pool, const string& nspace) {

  // namespace
  string sfxspace = "sfx.";
  if (!nspace.empty()) sfxspace = nspace + ".sfx.";
  string llspace = "lowlevel.";
  if (!nspace.empty()) llspace = nspace + ".lowlevel.";

  vector<Real> pitch = pool.value<vector<Real> >(llspace + "pitch");

  standard::Algorithm* maxtt = standard::AlgorithmFactory::create("MaxToTotal");
  Real maxToTotal;
  maxtt->input("envelope").set(pitch);
  maxtt->output("maxToTotal").set(maxToTotal);
  maxtt->compute();
  pool.set(sfxspace + "pitch_max_to_total", maxToTotal);

  standard::Algorithm* mintt = standard::AlgorithmFactory::create("MinToTotal");
  Real minToTotal;
  mintt->input("envelope").set(pitch);
  mintt->output("minToTotal").set(minToTotal);
  mintt->compute();
  pool.set(sfxspace + "pitch_min_to_total", minToTotal);

  standard::Algorithm* pc = standard::AlgorithmFactory::create("Centroid");
  Real centroid;
  pc->configure("range", (uint)pitch.size() - 1);
  pc->input("array").set(pitch);
  pc->output("centroid").set(centroid);
  pc->compute();
  pool.set(sfxspace + "pitch_centroid", centroid);

  standard::Algorithm* amt = standard::AlgorithmFactory::create("AfterMaxToBeforeMaxEnergyRatio");
  Real ratio;
  amt->input("pitch").set(pitch);
  amt->output("afterMaxToBeforeMaxEnergyRatio").set(ratio);
  amt->compute();
  pool.set(sfxspace + "pitch_after_max_to_before_max_energy_ratio", ratio);

  delete maxtt;
  delete mintt;
  delete pc;
  delete amt;
}

void TonalPoolCleaning(Pool& pool, const string& nspace) {

  // namespace
  string tonalspace = "tonal.";
  if (!nspace.empty()) tonalspace = nspace + ".tonal.";

  Real tuningFreq = pool.value<vector<Real> >(tonalspace + "tuning_frequency").back();
  pool.remove(tonalspace + "tuning_frequency");
  pool.set(tonalspace + "tuning_frequency", tuningFreq);

  // remove the highres hpcp which were only used to compute other features
  pool.remove(tonalspace + "hpcp_highres");
}


void PCA(Pool& pool, const string& nspace) {

  // namespace:
  string llspace = "lowlevel.";
  if (!nspace.empty()) llspace = nspace + ".lowlevel.";

  vector<vector<Real> > sccoeffs = pool.value<vector<vector<Real> > >(llspace + "sccoeffs");
  vector<vector<Real> > scvalleys = pool.value<vector<vector<Real> > >(llspace + "scvalleys");

  Pool poolSc, poolTransformed;

  for (int iFrame = 0; iFrame < (int)sccoeffs.size(); iFrame++) {
    vector<Real> merged(2*sccoeffs[iFrame].size(), 0.0);
    for(int i=0, j=0; i<(int)sccoeffs[iFrame].size(); i++, j++) {
      merged[j++] = sccoeffs[iFrame][i];
      merged[j] = scvalleys[iFrame][i];
    }
    poolSc.add("contrast", merged);
  }

  standard::Algorithm* pca  = standard::AlgorithmFactory::create("PCA",
                                             "namespaceIn",  "contrast",
                                             "namespaceOut", "contrast");
  pca->input("poolIn").set(poolSc);
  pca->output("poolOut").set(poolTransformed);
  pca->compute();

  pool.set(llspace + "spectral_contrast.mean",
           meanFrames(poolTransformed.value<vector<vector<Real> > >("contrast")));
  pool.set(llspace + "spectral_contrast.var",
           varianceFrames(poolTransformed.value<vector<vector<Real> > >("contrast")));

  // remove original data from spectral contrast:
  pool.remove(llspace + "sccoeffs");
  pool.remove(llspace + "scvalleys");
  delete pca;
}

// Add missing descriptors which are not computed yet, but will be for the
// final release or during the 1.x cycle. However, the schema need to be
// complete before that, so just put default values for these.
// Also make sure that some descriptors that might have fucked up come out nice.
void PostProcess(Pool& pool, const string& nspace) {
  string rhythmspace = "rhythm.";
  if (!nspace.empty()) rhythmspace = nspace + ".rhythm.";
  pool.set(rhythmspace + "bpm_confidence", 0.0);
  pool.set(rhythmspace + "perceptual_tempo", "unknown");

  try {
    pool.value<vector<Real> >(rhythmspace + "beats_loudness");
  }
  catch (EssentiaException&) {
    pool.set(rhythmspace + "beats_loudness", 0.0);
    pool.set(rhythmspace + "beats_loudness_bass", 0.0);
  }

  try {
    pool.value<vector<Real> >(rhythmspace + "rubato_start");
  }
  catch (EssentiaException&) {
    pool.set(rhythmspace + "rubato_start", vector<Real>(0));
  }

  try {
    pool.value<vector<Real> >(rhythmspace + "rubato_stop");
  }
  catch (EssentiaException&) {
    pool.set(rhythmspace + "rubato_stop", vector<Real>(0));
  }

  // PCA analysis of spectral contrast output:
  PCA(pool, nspace);
}

void getAnalysisData(const essentia::Pool& pool, Real& replayGain, Real& sampleRate, string& downmix) {
  // stores values for replayGain, sampleRate and downmix type
  // used to obtain the results in pool
  try {
    replayGain = pool.value<Real>("metadata.audio_properties.replay_gain");
  }
  catch(const EssentiaException&) {
    throw EssentiaException("Error: replay gain not found in pool but it is mandatory for computing descriptors");
  }
  try {
    sampleRate = pool.value<Real>("metadata.audio_properties.analysis_sample_rate");
  }
  catch(const EssentiaException&) {
    throw EssentiaException("Error: could not find analysis sampling rate in pool");
  }
  try {
    downmix = pool.value<string>("metadata.audio_properties.downmix");
  }
  catch (const EssentiaException&) {
    throw EssentiaException("Error: could not determine downmix type");
  }
}
