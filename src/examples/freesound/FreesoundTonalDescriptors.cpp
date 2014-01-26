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

#include "FreesoundTonalDescriptors.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;

const string FreesoundTonalDescriptors::nameSpace="tonal.";  


 void FreesoundTonalDescriptors ::createNetwork(SourceBase& source, Pool& pool){

  int frameSize = 8192;
  int hopSize =   4096;
  string silentFrames = "noise";
  string windowType = "blackmanharris92";
  int zeroPadding = 0;

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();


  // FrameCutter
  Algorithm* fc = factory.create("FrameCutter",
                                 "frameSize", frameSize,
                                 "hopSize", hopSize,
                                 "silentFrames", silentFrames);
  source >> fc->input("signal");

  // Windowing
  Algorithm* w = factory.create("Windowing",
                                "type", windowType,
                                "zeroPadding", zeroPadding);
  fc->output("frame") >> w->input("frame");

  // Spectrum
  Algorithm* spec = factory.create("Spectrum");
  w->output("frame") >> spec->input("frame");

  // Spectral Peaks
  Algorithm* peaks = factory.create("SpectralPeaks",
                                    "maxPeaks", 10000,
                                    "magnitudeThreshold", 0.00001,
                                    "minFrequency", 40,
                                    "maxFrequency", 5000,
                                    "orderBy", "magnitude");
  spec->output("spectrum") >> peaks->input("spectrum");

  // Tuning Frequency
  Algorithm* tuning = factory.create("TuningFrequency");
  peaks->output("magnitudes") >> tuning->input("magnitudes");
  peaks->output("frequencies") >> tuning->input("frequencies");
  tuning->output("tuningFrequency") >> PC(pool, nameSpace + "tuning_frequency");
  tuning->output("tuningCents") >> NOWHERE;

  // TODO: tuning frequency is currently provided but not used for HPCP computation
  Real tuningFreq = 440;

  
  Algorithm* hpcp_key = factory.create("HPCP",
                                       "size", 36,
                                       "referenceFrequency", tuningFreq,
                                       "bandPreset", false,
                                       "minFrequency", 40.0,
                                       "maxFrequency", 5000.0,
                                       "weightType", "squaredCosine",
                                       "nonLinear", false,
                                       "windowSize", 4.0/3.0);
  peaks->output("frequencies") >> hpcp_key->input("frequencies");
  peaks->output("magnitudes") >> hpcp_key->input("magnitudes");
  hpcp_key->output("hpcp") >> PC(pool, nameSpace + "hpcp");

  
  Algorithm* skey = factory.create("Key");
  hpcp_key->output("hpcp") >> skey->input("pcp");
  skey->output("key") >> PC(pool, nameSpace + "key_key");
  skey->output("scale") >> PC(pool, nameSpace + "key_scale");
  skey->output("strength") >> PC(pool, nameSpace + "key_strength");

  
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
  peaks->output("frequencies") >> hpcp_chord->input("frequencies");
  peaks->output("magnitudes") >> hpcp_chord->input("magnitudes");


  Algorithm* schord = factory.create("ChordsDetection");
  hpcp_chord->output("hpcp") >> schord->input("pcp");
  schord->output("chords") >> PC(pool, nameSpace + "chords_progression");
  schord->output("strength") >> PC(pool, nameSpace + "chords_strength");

  
  Algorithm* schords_desc = factory.create("ChordsDescriptors");
  schord->output("chords") >> schords_desc->input("chords");
  skey->output("key") >> schords_desc->input("key");
  skey->output("scale") >> schords_desc->input("scale");

  schords_desc->output("chordsHistogram") >> PC(pool, nameSpace + "chords_histogram");
  schords_desc->output("chordsNumberRate") >> PC(pool, nameSpace + "chords_number_rate");
  schords_desc->output("chordsChangesRate") >> PC(pool, nameSpace + "chords_changes_rate");
  schords_desc->output("chordsKey") >> PC(pool, nameSpace + "chords_key");
  schords_desc->output("chordsScale") >> PC(pool, nameSpace + "chords_scale");
 }
