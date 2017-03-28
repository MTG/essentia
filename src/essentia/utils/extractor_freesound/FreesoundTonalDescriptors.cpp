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

#include "FreesoundTonalDescriptors.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;

const string FreesoundTonalDescriptors::nameSpace="tonal.";

void FreesoundTonalDescriptors ::createNetwork(SourceBase& source, Pool& pool) {

  int frameSize = int(options.value<Real>("tonal.frameSize"));
  int hopSize =   int(options.value<Real>("tonal.hopSize"));
  string silentFrames = options.value<string>("tonal.silentFrames");
  string windowType = options.value<string>("tonal.windowType");
  int zeroPadding = int(options.value<Real>("tonal.zeroPadding"));

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* fc = factory.create("FrameCutter",
                                 "frameSize", frameSize,
                                 "hopSize", hopSize,
                                 "silentFrames", silentFrames);
  Algorithm* w = factory.create("Windowing",
                                "type", windowType,
                                "zeroPadding", zeroPadding);
  Algorithm* spec = factory.create("Spectrum");

  // Compute tuning frequency
  Algorithm* peaks = factory.create("SpectralPeaks",
                                    "maxPeaks", 10000,
                                    "magnitudeThreshold", 0.00001,
                                    "minFrequency", 40,
                                    "maxFrequency", 5000,
                                    "orderBy", "magnitude");
  Algorithm* tuning = factory.create("TuningFrequency");

  source >> fc->input("signal");
  fc->output("frame") >> w->input("frame");
  w->output("frame") >> spec->input("frame");
  spec->output("spectrum") >> peaks->input("spectrum");
  peaks->output("magnitudes") >> tuning->input("magnitudes");
  peaks->output("frequencies") >> tuning->input("frequencies");
  tuning->output("tuningFrequency") >> PC(pool, nameSpace + "tuning_frequency");
  tuning->output("tuningCents") >> NOWHERE;

  // Compute HPCP and key

  // TODO: Tuning frequency is currently provided but not used for HPCP 
  // computation, not clear if it would make an improvement for freesound sounds
  Real tuningFreq = 440;
  //Real tuningFreq = pool.value<vector<Real> >(nameSpace + "tuning_frequency").back();

  Algorithm* hpcp_peaks = factory.create("SpectralPeaks",
                                         "maxPeaks", 60,
                                         "magnitudeThreshold", 0.00001,
                                         "minFrequency", 20.0,
                                         "maxFrequency", 3500.0,
                                         "orderBy", "magnitude");
  // This is taken from MusicExtractor: Detecting 60 peaks instead of all of 
  // all peaks may be better, especially for electronic music that has lots of 
  // high-frequency content

  Algorithm* hpcp_key = factory.create("HPCP",
                                       "size", 36,
                                       "referenceFrequency", tuningFreq,
                                       "bandPreset", false,
                                       "minFrequency", 20.0,
                                       "maxFrequency", 3500.0,
                                       "weightType", "cosine",
                                       "nonLinear", false,
                                       "windowSize", 1.);
  // Previously used parameter values: 
  // - nonLinear = false
  // - weightType = squaredCosine
  // - windowSize = 4.0/3.0
  // - bandPreset = false
  // - minFrequency = 40
  // - maxFrequency = 5000

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

  spec->output("spectrum") >> hpcp_peaks->input("spectrum");
  hpcp_peaks->output("frequencies") >> hpcp_key->input("frequencies");
  hpcp_peaks->output("magnitudes")  >> hpcp_key->input("magnitudes");

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

  // Compute chords
  // TODO review these parameters to improve chords detection. Keeping old code for now
  Algorithm* hpcp_chord = factory.create("HPCP",
                                         "size", 36,
                                         "referenceFrequency", tuningFreq,
                                         "harmonics", 8,
                                         "bandPreset", true,
                                         "minFrequency", 40.0,
                                         "maxFrequency", 5000.0,
                                         "bandSplitFrequency", 500.0,
                                         "weightType", "cosine",
                                         "nonLinear", true,
                                         "windowSize", 0.5);
  peaks->output("frequencies") >> hpcp_chord->input("frequencies");
  peaks->output("magnitudes") >> hpcp_chord->input("magnitudes");

  Algorithm* schord = factory.create("ChordsDetection");
  hpcp_chord->output("hpcp") >> schord->input("pcp");
  schord->output("chords") >> PC(pool, nameSpace + "chords_progression");
  schord->output("strength") >> PC(pool, nameSpace + "chords_strength");
  
  Algorithm* schords_desc = factory.create("ChordsDescriptors");
  schord->output("chords") >> schords_desc->input("chords");
  skey_temperley->output("key") >> schords_desc->input("key");
  skey_temperley->output("scale") >> schords_desc->input("scale");

  schords_desc->output("chordsHistogram") >> PC(pool, nameSpace + "chords_histogram");
  schords_desc->output("chordsNumberRate") >> PC(pool, nameSpace + "chords_number_rate");
  schords_desc->output("chordsChangesRate") >> PC(pool, nameSpace + "chords_changes_rate");
  schords_desc->output("chordsKey") >> PC(pool, nameSpace + "chords_key");
  schords_desc->output("chordsScale") >> PC(pool, nameSpace + "chords_scale");
     
  Algorithm* entropy = factory.create("Entropy");
  hpcp_chord->output("hpcp") >> entropy->input("array");
  entropy->output("entropy") >> PC(pool, nameSpace + "hpcp_entropy");

  Algorithm* crest = factory.create("Crest");
  hpcp_chord->output("hpcp") >> crest->input("array");
  crest->output("crest") >> PC(pool, nameSpace + "hpcp_crest");
}
