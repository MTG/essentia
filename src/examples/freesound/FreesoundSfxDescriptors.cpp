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


#include "FreesoundSfxDescriptors.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;


using namespace std;

const string FreesoundSfxDescriptors::nameSpace="sfx.";  

  // TODO: normalization of centroids requires array size or duration, 
  // not directly available in streaming mode.
  // Can be done at the client for the moment

void  FreesoundSfxDescriptors::createNetwork(SourceBase& source, Pool& pool){
  
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  // Envelope
  Algorithm* envelope = factory.create("Envelope");
  source >> envelope->input("signal");

  // Temporal Statistics
  Algorithm* decrease = factory.create("Decrease");
  Algorithm* accu = factory.create("RealAccumulator");
  envelope->output("signal") >> accu->input("data");
  accu->output("array") >> decrease->input("array");
  decrease->output("decrease") >> PC(pool, nameSpace + "temporal_decrease");

  // Audio Central Moments Statistics
  Algorithm* cm = factory.create("CentralMoments");
  Algorithm* ds = factory.create("DistributionShape");
  accu->output("array") >> cm->input("array");
  cm->output("centralMoments") >> ds->input("centralMoments");
  ds->output("kurtosis") >> PC(pool, nameSpace + "temporal_kurtosis");
  ds->output("spread") >> PC(pool, nameSpace + "temporal_spread");
  ds->output("skewness") >> PC(pool, nameSpace + "temporal_skewness");

  // Temporal Centroid
  Algorithm* centroid = factory.create("Centroid");
  accu->output("array") >> centroid->input("array");
  centroid->output("centroid") >> PC(pool, nameSpace + "temporal_centroid");

  // Effective Duration
  Algorithm* duration = factory.create("EffectiveDuration");
  accu->output("array") >> duration->input("signal");
  duration->output("effectiveDuration") >> PC(pool, nameSpace + "effective_duration");

  // Log Attack Time
  Algorithm* log = factory.create("LogAttackTime");
  accu->output("array") >> log->input("signal");
  log->output("logAttackTime") >> PC(pool, nameSpace + "logattacktime");

  // Strong Decay
  Algorithm* decay = factory.create("StrongDecay");
  envelope->output("signal") >> decay->input("signal");
  decay->output("strongDecay") >> PC(pool, nameSpace + "strongdecay");

  // Flatness
  Algorithm* flatness = factory.create("FlatnessSFX");
  accu->output("array") >> flatness->input("envelope");
  flatness->output("flatness") >> PC(pool, nameSpace + "flatness");

  // Morphological Descriptors
  Algorithm* max1 = factory.create("MaxToTotal");
  envelope->output("signal") >> max1->input("envelope");
  max1->output("maxToTotal") >> PC(pool, nameSpace + "max_to_total");

  Algorithm* tc = factory.create("TCToTotal");
  envelope->output("signal") >> tc->input("envelope");
  tc->output("TCToTotal") >> PC(pool, nameSpace + "tc_to_total");

  Algorithm* der = factory.create("DerivativeSFX");
  accu->output("array") >> der->input("envelope");
  der->output("derAvAfterMax") >> PC(pool, nameSpace + "der_av_after_max");
  der->output("maxDerBeforeMax") >>  PC(pool, nameSpace + "max_der_before_max");
}




void  FreesoundSfxDescriptors::createPitchNetwork(VectorInput<Real>& pitch, Pool& pool){

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* maxtt = factory.create("MaxToTotal");
  pitch >> maxtt->input("envelope"); // TODO: should we use Envelope?
  maxtt->output("maxToTotal") >>  PC(pool, nameSpace + "pitch_max_to_total");

  Algorithm* mintt = factory.create("MinToTotal");
  pitch >> mintt->input("envelope"); // TODO: should we use Envelope?
  mintt->output("minToTotal") >>  PC(pool, nameSpace + "pitch_min_to_total");

  Algorithm* accu = factory.create("RealAccumulator");
  pitch >> accu->input("data"); 

  Algorithm* pc = factory.create("Centroid");

  accu->output("array") >> pc->input("array"); 
  pc->output("centroid") >>  PC(pool, nameSpace + "pitch_centroid");
  

  Algorithm* amt = factory.create("AfterMaxToBeforeMaxEnergyRatio");
  pitch >> amt->input("pitch"); 
  amt->output("afterMaxToBeforeMaxEnergyRatio") >>  PC(pool, nameSpace + "pitch_after_max_to_before_max_energy_ratio");
  
}



void  FreesoundSfxDescriptors::createHarmonicityNetwork(SourceBase& source, Pool& pool){

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  //Real sampleRate = 44100;
  int frameSize =   2048;
  int hopSize =     1024;
  int zeroPadding = 0;
  string silentFrames ="noise";
  string windowType = "blackmanharris62";


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

  Algorithm* harmPeaks = factory.create("HarmonicPeaks");
  Algorithm* peaks = factory.create("SpectralPeaks",
                                    "orderBy", "frequency","minFrequency", 20);
  spec->output("spectrum") >> peaks->input("spectrum");

  // Pitch Detection
  Algorithm* pitch = factory.create("PitchYinFFT",
                                    "frameSize", frameSize);
  spec->output("spectrum") >> pitch->input("spectrum");
  pitch->output("pitchConfidence") >> NOWHERE;

  peaks->output("frequencies") >> harmPeaks->input("frequencies");
  
  peaks->output("magnitudes") >> harmPeaks->input("magnitudes");
  pitch->output("pitch") >> harmPeaks->input("pitch");

  Algorithm* odd2even = factory.create("OddToEvenHarmonicEnergyRatio");
  Algorithm* tristimulus = factory.create("Tristimulus");
  Algorithm* inharmonicity = factory.create("Inharmonicity");

  // inputs
  harmPeaks->output("harmonicFrequencies") >> tristimulus->input("frequencies");
  harmPeaks->output("harmonicMagnitudes")  >>  tristimulus->input("magnitudes");
  harmPeaks->output("harmonicFrequencies") >> odd2even->input("frequencies");
  harmPeaks->output("harmonicMagnitudes")  >> odd2even->input("magnitudes");
  harmPeaks->output("harmonicFrequencies") >> inharmonicity->input("frequencies");
  harmPeaks->output("harmonicMagnitudes")  >> inharmonicity->input("magnitudes");

  // outputs
  inharmonicity->output("inharmonicity") >> PC(pool, nameSpace + "inharmonicity");
  odd2even->output("oddToEvenHarmonicEnergyRatio") >> PC(pool, nameSpace + "oddtoevenharmonicenergyratio");
  tristimulus->output("tristimulus") >> PC(pool, nameSpace + "tristimulus");
}