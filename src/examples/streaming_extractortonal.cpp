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

#include "streaming_extractortonal.h"
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/streaming/algorithms/poolstorage.h>

using namespace std;
using namespace essentia;
using namespace essentia::streaming;

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

void TuningFrequency(SourceBase& input, Pool& pool, const Pool& options, const string& nspace) {

  // namespace
  string tonalspace = "tonal.";
  if (!nspace.empty()) tonalspace = nspace + ".tonal.";

  int frameSize = int(options.value<Real>("tonal.frameSize"));
  int hopSize =   int(options.value<Real>("tonal.hopSize"));
  string silentFrames = options.value<string>("tonal.silentFrames");
  string windowType = options.value<string>("tonal.windowType");
  int zeroPadding = int(options.value<Real>("tonal.zeroPadding"));

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  // FrameCutter
  Algorithm* fc = factory.create("FrameCutter",
                                 "frameSize", frameSize,
                                 "hopSize", hopSize,
                                 "silentFrames", silentFrames);
  connect(input, fc->input("signal"));

  // Windowing
  Algorithm* w = factory.create("Windowing",
                                "type", windowType,
                                "zeroPadding", zeroPadding);
  connect(fc->output("frame"), w->input("frame"));

  // Spectrum
  Algorithm* spec = factory.create("Spectrum");
  connect(w->output("frame"), spec->input("frame"));

  // Spectral Peaks
  Algorithm* peaks = factory.create("SpectralPeaks",
                                    "maxPeaks", 10000,
                                    "magnitudeThreshold", 0.00001,
                                    "minFrequency", 40,
                                    "maxFrequency", 5000,
                                    "orderBy", "frequency");
  connect(spec->output("spectrum"), peaks->input("spectrum"));

  // Tuning Frequency
  Algorithm* tuning = factory.create("TuningFrequency");
  connect(peaks->output("magnitudes"), tuning->input("magnitudes"));
  connect(peaks->output("frequencies"), tuning->input("frequencies"));
  connect(tuning->output("tuningFrequency"), pool, tonalspace + "tuning_frequency");
  connect(tuning->output("tuningCents"), NOWHERE);

}

void TonalDescriptors(SourceBase& input, Pool& pool, const Pool& options, const string& nspace) {

  // namespace
  string tonalspace = "tonal.";
  if (!nspace.empty()) tonalspace = nspace + ".tonal.";

  int frameSize = int(options.value<Real>("tonal.frameSize"));
  int hopSize =   int(options.value<Real>("tonal.hopSize"));
  string silentFrames = options.value<string>("tonal.silentFrames");
  string windowType = options.value<string>("tonal.windowType");
  int zeroPadding = int(options.value<Real>("tonal.zeroPadding"));

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  // FrameCutter
  Algorithm* fc = factory.create("FrameCutter",
                                 "frameSize", frameSize,
                                 "hopSize", hopSize,
                                 "silentFrames", silentFrames);
  connect(input, fc->input("signal"));

  // Windowing
  Algorithm* w = factory.create("Windowing",
                                "type", windowType,
                                "zeroPadding", zeroPadding);
  connect(fc->output("frame"), w->input("frame"));

  // Spectrum
  Algorithm* spec = factory.create("Spectrum");
  connect(w->output("frame"), spec->input("frame"));

  // Spectral Peaks
  Algorithm* peaks = factory.create("SpectralPeaks",
                                    "maxPeaks", 10000,
                                    "magnitudeThreshold", 0.00001,
                                    "minFrequency", 40,
                                    "maxFrequency", 5000,
                                    "orderBy", "magnitude");
  connect(spec->output("spectrum"), peaks->input("spectrum"));

  // Tuning Frequency
  Real tuningFreq = pool.value<vector<Real> >(tonalspace + "tuning_frequency").back();

  // HPCP Key
  Algorithm* hpcp_key = factory.create("HPCP",
                                       "size", 36,
                                       "referenceFrequency", tuningFreq,
                                       "bandPreset", false,
                                       "minFrequency", 40.0,
                                       "maxFrequency", 5000.0,
                                       "weightType", "squaredCosine",
                                       "nonLinear", false,
                                       "windowSize", 4.0/3.0);
  connect(peaks->output("frequencies"), hpcp_key->input("frequencies"));
  connect(peaks->output("magnitudes"), hpcp_key->input("magnitudes"));
  connect(hpcp_key->output("hpcp"), pool, tonalspace + "hpcp");

  // native streaming Key algo
  Algorithm* skey = factory.create("Key");
  connect(hpcp_key->output("hpcp"), skey->input("pcp"));
  connect(skey->output("key"), pool, tonalspace + "key_key");
  connect(skey->output("scale"), pool, tonalspace + "key_scale");
  connect(skey->output("strength"), pool, tonalspace + "key_strength");

  // HPCP Chord
  Algorithm* hpcp_chord = factory.create("HPCP",
                                         "size", 36,
                                         "referenceFrequency", tuningFreq,
                                         "harmonics", 8,
                                         "bandPreset", true,
                                         "minFrequency", 40.0,
                                         "maxFrequency", 5000.0,
                                         "splitFrequency", 500.0,
                                         "weightType", "cosine",
                                         "nonLinear", true,
                                         "windowSize", 0.5);
  connect(peaks->output("frequencies"), hpcp_chord->input("frequencies"));
  connect(peaks->output("magnitudes"), hpcp_chord->input("magnitudes"));

  // native streaming chords algo
  Algorithm* schord = factory.create("ChordsDetection");
  connect(hpcp_chord->output("hpcp"), schord->input("pcp"));
  connect(schord->output("chords"), pool, tonalspace + "chords_progression");
  connect(schord->output("strength"), pool, tonalspace + "chords_strength");

  // native streaming chords descriptors algo
  Algorithm* schords_desc = factory.create("ChordsDescriptors");
  connect(schord->output("chords"), schords_desc->input("chords"));
  connect(skey->output("key"), schords_desc->input("key"));
  connect(skey->output("scale"), schords_desc->input("scale"));

  connect(schords_desc->output("chordsHistogram"), pool, tonalspace + "chords_histogram");
  connect(schords_desc->output("chordsNumberRate"), pool, tonalspace + "chords_number_rate");
  connect(schords_desc->output("chordsChangesRate"), pool, tonalspace + "chords_changes_rate");
  connect(schords_desc->output("chordsKey"), pool, tonalspace + "chords_key");
  connect(schords_desc->output("chordsScale"), pool, tonalspace + "chords_scale");

  // HPCP Tuning
  Algorithm* hpcp_tuning = factory.create("HPCP",
                                          "size", 120,
                                          "referenceFrequency", tuningFreq,
                                          "harmonics", 8,
                                          "bandPreset", true,
                                          "minFrequency", 40.0,
                                          "maxFrequency", 5000.0,
                                          "splitFrequency", 500.0,
                                          "weightType", "cosine",
                                          "nonLinear", true,
                                          "windowSize", 0.5);
  connect(peaks->output("frequencies"), hpcp_tuning->input("frequencies"));
  connect(peaks->output("magnitudes"), hpcp_tuning->input("magnitudes"));

  connect(hpcp_tuning->output("hpcp"), pool, tonalspace + "hpcp_highres");
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
