/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include "MusicTonalDescriptors.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;

const string MusicTonalDescriptors::nameSpace="tonal.";  

void MusicTonalDescriptors::createNetworkTuningFrequency(SourceBase& source, Pool& pool){

  int frameSize = int(options.value<Real>("tonal.frameSize"));
  int hopSize =   int(options.value<Real>("tonal.hopSize"));
  string silentFrames = options.value<string>("tonal.silentFrames");
  string windowType = options.value<string>("tonal.windowType");
  int zeroPadding = int(options.value<Real>("tonal.zeroPadding"));

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* fc     = factory.create("FrameCutter",
                                     "frameSize", frameSize,
                                     "hopSize", hopSize,
                                     "silentFrames", silentFrames);
  Algorithm* w      = factory.create("Windowing",
                                     "type", windowType,
                                     "zeroPadding", zeroPadding);
  Algorithm* spec   = factory.create("Spectrum");
  // TODO: which parameters to select for min/maxFrequency? [20, 3500] for consistency?
  Algorithm* peaks  = factory.create("SpectralPeaks",
                                     "maxPeaks", 10000,
                                     "magnitudeThreshold", 0.00001,
                                     "minFrequency", 40,
                                     "maxFrequency", 5000,
                                     "orderBy", "frequency");
  Algorithm* tuning = factory.create("TuningFrequency");

  source                            >> fc->input("signal");
  fc->output("frame")               >> w->input("frame");
  w->output("frame")                >> spec->input("frame");
  spec->output("spectrum")          >> peaks->input("spectrum");
  peaks->output("magnitudes")       >> tuning->input("magnitudes");
  peaks->output("frequencies")      >> tuning->input("frequencies");
  tuning->output("tuningFrequency") >> PC(pool, nameSpace + "tuning_frequency");
  tuning->output("tuningCents")     >> NOWHERE;
}

void MusicTonalDescriptors::createNetwork(SourceBase& source, Pool& pool) {

  // Using the 20-3500 Hz frequency ranges as suggested by Angel Faraldo
  // This range improves key estimation significantly for electronic music 
  // (lowering max frequency removes harmonics that would confuse  estimation).
  // It does not degrate estimation for pop music. Not evaluated on classical.
  // Previous range: 40-5000 Hz

  int frameSize = int(options.value<Real>("tonal.frameSize"));
  int hopSize =   int(options.value<Real>("tonal.hopSize"));
  string silentFrames = options.value<string>("tonal.silentFrames");
  string windowType = options.value<string>("tonal.windowType");
  int zeroPadding = int(options.value<Real>("tonal.zeroPadding"));

  Real tuningFreq = pool.value<vector<Real> >(nameSpace + "tuning_frequency").back();

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* fc = factory.create("FrameCutter",
                                 "frameSize", frameSize,
                                 "hopSize", hopSize,
                                 "silentFrames", silentFrames);
  Algorithm* w = factory.create("Windowing",
                                "type", windowType,
                                "zeroPadding", zeroPadding);
  Algorithm* spec = factory.create("Spectrum");
  Algorithm* peaks = factory.create("SpectralPeaks",
                                    "maxPeaks", 60,
                                    "magnitudeThreshold", 0.00001,
                                    "minFrequency", 20.0,
                                    "maxFrequency", 3500.0,
                                    "orderBy", "magnitude");
  // Detecting 60 peaks instead of all of them as it may be better not to 
  // consider too many harmonics, especially for electronic music

  // Using HPCP parameters recommended for electronic music:
  Algorithm* hpcp_key = factory.create("HPCP",
                                       "size", 36,
                                       "referenceFrequency", tuningFreq,
                                       "bandPreset", false,
                                       "minFrequency", 20.0,
                                       //"bandSplitFrequency", 250.0, // TODO ???
                                       "maxFrequency", 3500.0,
                                       "weightType", "cosine",
                                       "nonLinear", false,
                                       "windowSize", 1.);
  // Previously used parameter values: 
  // - nonLinear = false
  // - weightType = squaredCosine
  // - windowSize = 4.0/3.0
  // - bandPreset = true

  Algorithm* skey_temperley = factory.create("Key",
                                   "numHarmonics", 4,
                                   "pcpSize", 36,
                                   "profileType", "temperley",
                                   "slope", 0.6,
                                   "usePolyphony", true,
                                   "useThreeChords", true);

  Algorithm* skey_krumhansl = factory.create("Key",
                                   "numHarmonics", 4,
                                   "pcpSize", 36,
                                   "profileType", "krumhansl",
                                   "slope", 0.6,
                                   "usePolyphony", true,
                                   "useThreeChords", true);

  Algorithm* skey_edma = factory.create("Key",
                                   "numHarmonics", 4,
                                   "pcpSize", 36,
                                   "profileType", "edma",
                                   "slope", 0.6,
                                   "usePolyphony", true,
                                   "useThreeChords", true);

  // TODO review this parameters to improve our chords detection
  Algorithm* hpcp_chord = factory.create("HPCP",
                                         "size", 36,
                                         "referenceFrequency", tuningFreq,
                                         "harmonics", 8,
                                         "bandPreset", true,
                                         "minFrequency", 20.0,
                                         "maxFrequency", 3500.0,
                                         "bandSplitFrequency", 500.0,
                                         "weightType", "cosine",
                                         "nonLinear", true,
                                         "windowSize", 0.5);

  Algorithm* schord = factory.create("ChordsDetection");
  Algorithm* schords_desc = factory.create("ChordsDescriptors");

  source                       >> fc->input("signal");
  fc->output("frame")          >> w->input("frame");
  w->output("frame")           >> spec->input("frame");
  spec->output("spectrum")     >> peaks->input("spectrum");

  peaks->output("frequencies") >> hpcp_key->input("frequencies");
  peaks->output("magnitudes")  >> hpcp_key->input("magnitudes");
  hpcp_key->output("hpcp")     >> PC(pool, nameSpace + "hpcp");
  hpcp_key->output("hpcp")     >> skey_temperley->input("pcp");
  hpcp_key->output("hpcp")     >> skey_krumhansl->input("pcp");
  hpcp_key->output("hpcp")     >> skey_edma->input("pcp");

  skey_temperley->output("key")          >> PC(pool, nameSpace + "key_temperley.key");
  skey_temperley->output("scale")        >> PC(pool, nameSpace + "key_temperley.scale");
  skey_temperley->output("strength")     >> PC(pool, nameSpace + "key_temperley.strength");

  skey_krumhansl->output("key")          >> PC(pool, nameSpace + "key_krumhansl.key");
  skey_krumhansl->output("scale")        >> PC(pool, nameSpace + "key_krumhansl.scale");
  skey_krumhansl->output("strength")     >> PC(pool, nameSpace + "key_krumhansl.strength");

  skey_edma->output("key")          >> PC(pool, nameSpace + "key_edma.key");
  skey_edma->output("scale")        >> PC(pool, nameSpace + "key_edma.scale");
  skey_edma->output("strength")     >> PC(pool, nameSpace + "key_edma.strength");


  peaks->output("frequencies") >> hpcp_chord->input("frequencies");
  peaks->output("magnitudes")  >> hpcp_chord->input("magnitudes");
  hpcp_chord->output("hpcp")   >> schord->input("pcp");
  schord->output("strength")   >> PC(pool, nameSpace + "chords_strength");
  
  // TODO: Chords progression has low practical sense and is based on a very simple algorithm prone to errors.
  // We need to have better algorithm first to include this descriptor.
  // schord->output("chords") >> PC(pool, nameSpace + "chords_progression");
  
  // TODO: chord histogram is aligned so than the first bin is the overall key
  //       estimated using temperley profile. Should we align to the most 
  //       frequent chord instead?  
  schord->output("chords")               >> schords_desc->input("chords");
  skey_temperley->output("key")          >> schords_desc->input("key");
  skey_temperley->output("scale")        >> schords_desc->input("scale");
  schords_desc->output("chordsHistogram")   >> PC(pool, nameSpace + "chords_histogram");
  schords_desc->output("chordsNumberRate")  >> PC(pool, nameSpace + "chords_number_rate");
  schords_desc->output("chordsChangesRate") >> PC(pool, nameSpace + "chords_changes_rate");
  schords_desc->output("chordsKey")         >> PC(pool, nameSpace + "chords_key");
  schords_desc->output("chordsScale")       >> PC(pool, nameSpace + "chords_scale");

  // HPCP Entropy and Crest
  Algorithm* ent = factory.create("Entropy");
  hpcp_chord->output("hpcp")  >> ent->input("array");
  ent->output("entropy")      >> PC(pool, nameSpace + "hpcp_entropy");
  
  Algorithm* crest = factory.create("Crest");
  hpcp_chord->output("hpcp") >> crest->input("array");
  crest->output("crest") >> PC(pool, nameSpace + "hpcp_crest");

  // HPCP Tuning
  Algorithm* hpcp_tuning = factory.create("HPCP",
                                          "size", 120,
                                          "referenceFrequency", tuningFreq,
                                          "harmonics", 8,
                                          "bandPreset", true,
                                          "minFrequency", 20.0,
                                          "maxFrequency", 3500.0,
                                          "bandSplitFrequency", 500.0,
                                          "weightType", "cosine",
                                          "nonLinear", true,
                                          "windowSize", 0.5);

  peaks->output("frequencies")  >> hpcp_tuning->input("frequencies");
  peaks->output("magnitudes")   >> hpcp_tuning->input("magnitudes");
  hpcp_tuning->output("hpcp")   >> PC(pool, nameSpace + "hpcp_highres");
}


void MusicTonalDescriptors::computeTuningSystemFeatures(Pool& pool){

  vector<Real> hpcp_highres = meanFrames(pool.value<vector<vector<Real> > >(nameSpace + "hpcp_highres"));
  pool.remove(nameSpace + "hpcp_highres");
  normalize(hpcp_highres);

  // 1- diatonic strength
  standard::AlgorithmFactory& factory = standard::AlgorithmFactory::instance();

  standard::Algorithm* keyDetect = factory.create("Key",
                                                  "numHarmonics", 4,
                                                  "pcpSize", 36,
                                                  "profileType", "diatonic",
                                                  "slope", 0.6,
                                                  "usePolyphony", true,
                                                  "useThreeChords", true);

  string key, scale;
  Real strength, unused;
  keyDetect->input("pcp").set(hpcp_highres);
  keyDetect->output("key").set(key);
  keyDetect->output("scale").set(scale);
  keyDetect->output("strength").set(strength);
  keyDetect->output("firstToSecondRelativeStrength").set(unused);
  keyDetect->compute();

  pool.set(nameSpace + "tuning_diatonic_strength", strength);

  // 2- high resolution features
  standard::Algorithm* highres = factory.create("HighResolutionFeatures");

  Real eqTempDeviation, ntEnergy, ntPeaks;
  highres->input("hpcp").set(hpcp_highres);
  highres->output("equalTemperedDeviation").set(eqTempDeviation);
  highres->output("nonTemperedEnergyRatio").set(ntEnergy);
  highres->output("nonTemperedPeaksEnergyRatio").set(ntPeaks);
  highres->compute();

  pool.set(nameSpace + "tuning_equal_tempered_deviation", eqTempDeviation);
  pool.set(nameSpace + "tuning_nontempered_energy_ratio", ntEnergy);

  // 3- THPCP
  vector<Real> hpcp = meanFrames(pool.value<vector<vector<Real> > >(nameSpace + "hpcp"));
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

  pool.set(nameSpace + "thpcp", hpcp);

  delete keyDetect;
  delete highres;
}
