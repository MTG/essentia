/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

  // Compute HPCP
  // NOTE: HPCP is no longer directly used for key extraction because we use KeyExtractor
  // which does the HPCP computation inside. Nevertheless, we keep HPCP computation here
  // so it can be added to the POOL.
  // NOTE: Tuning frequency is currently provided but not used for HPCP 
  // computation, not clear if it would make an improvement for Freesound sounds
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
  spec->output("spectrum") >> hpcp_peaks->input("spectrum");
  hpcp_peaks->output("frequencies") >> hpcp_key->input("frequencies");
  hpcp_peaks->output("magnitudes")  >> hpcp_key->input("magnitudes");
  hpcp_key->output("hpcp")     >> PC(pool, nameSpace + "hpcp");

  // Compute key
  Algorithm* key_extractor = factory.create("KeyExtractor",
                                           "profileType", "bgate",
                                           "frameSize", frameSize,
                                           "hopSize", frameSize,  // No overlapp (that's how KeyExtractor sets the defaults)
                                           "hpcpSize", 12);
  source >> key_extractor->input("audio");
  key_extractor->output("key") >> PC(pool, nameSpace + "key.key");
  key_extractor->output("scale") >> PC(pool, nameSpace + "key.scale");
  key_extractor->output("strength") >> PC(pool, nameSpace + "key.strength");

  Algorithm* entropy = factory.create("Entropy");
  hpcp_key->output("hpcp") >> entropy->input("array");
  entropy->output("entropy") >> PC(pool, nameSpace + "hpcp_entropy");

  Algorithm* crest = factory.create("Crest");
  hpcp_key->output("hpcp") >> crest->input("array");
  crest->output("crest") >> PC(pool, nameSpace + "hpcp_crest");
}
