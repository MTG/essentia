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


#include "MusicRhythmDescriptors.h"
using namespace std;
using namespace essentia;
using namespace streaming;


const string MusicRhythmDescriptors::nameSpace="rhythm."; 

void  MusicRhythmDescriptors::createNetwork(SourceBase& source, Pool& pool){
  
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  // Rhythm extractor
  Algorithm* rhythmExtractor = factory.create("RhythmExtractor2013");
  rhythmExtractor->configure("method", options.value<string>("rhythm.method"),
                             "maxTempo", (int) options.value<Real>("rhythm.maxTempo"),
                             "minTempo", (int) options.value<Real>("rhythm.minTempo"));

  source                                >> rhythmExtractor->input("signal");
  rhythmExtractor->output("ticks")      >> PC(pool, nameSpace + "beats_position");
  rhythmExtractor->output("bpm")        >> PC(pool, nameSpace + "bpm");
  rhythmExtractor->output("confidence") >> NOWHERE; 
  rhythmExtractor->output("estimates")  >> NOWHERE;
  // dummy "confidence" because 'degara' method does not estimate confidence
  // NOTE: we do not need bpm estimates and intervals in the pool because
  //       they can be deduced from ticks position and occupy too much space

  // BPM Histogram descriptors
  Algorithm* bpmhist = factory.create("BpmHistogramDescriptors");
  rhythmExtractor->output("bpmIntervals") >> bpmhist->input("bpmIntervals");

  // connect as single value otherwise PoolAggregator will compute statistics
  connectSingleValue(bpmhist->output("firstPeakBPM"), pool, nameSpace + "bpm_histogram_first_peak_bpm");
  connectSingleValue(bpmhist->output("firstPeakWeight"), pool, nameSpace + "bpm_histogram_first_peak_weight");
  connectSingleValue(bpmhist->output("firstPeakSpread"), pool, nameSpace + "bpm_histogram_first_peak_weight");
  connectSingleValue(bpmhist->output("secondPeakBPM"), pool, nameSpace + "bpm_histogram_second_peak_bpm");
  connectSingleValue(bpmhist->output("secondPeakWeight"), pool, nameSpace + "bpm_histogram_second_peak_weight");
  connectSingleValue(bpmhist->output("secondPeakSpread"), pool, nameSpace + "bpm_histogram_second_peak_spread");
  connectSingleValue(bpmhist->output("histogram"), pool, nameSpace + "bpm_histogram");

  // Onset Detection
  // TODO: use SuperFlux onset rate algorithm instead!
  //       the algorithm that is used is rather outdated, onset times can be 
  //       inaccurate, however, onset_rate is still very informative for many 
  //       tasks 
  Algorithm* onset = factory.create("OnsetRate");
  source                      >> onset->input("signal");
  onset->output("onsetTimes") >> NOWHERE;
  onset->output("onsetRate")  >> PC(pool, nameSpace + "onset_rate");

  // Danceability
  Algorithm* danceability = factory.create("Danceability");
  source                                >> danceability->input("signal");
  danceability->output("danceability")  >> PC(pool, nameSpace + "danceability");
  danceability->output("dfa")           >> NOWHERE;
}

void MusicRhythmDescriptors::createNetworkBeatsLoudness(SourceBase& source, Pool& pool){
 
  Real sampleRate = options.value<Real>("analysisSampleRate");

  AlgorithmFactory& factory = AlgorithmFactory::instance();
  vector<Real> ticks = pool.value<vector<Real> >(nameSpace + "beats_position");
  Algorithm* beatsLoudness = factory.create("BeatsLoudness",
                                            "sampleRate", sampleRate,
                                            "beats", ticks);
  source                                      >> beatsLoudness->input("signal");
  beatsLoudness->output("loudness")           >> PC(pool, nameSpace + "beats_loudness");
  beatsLoudness->output("loudnessBandRatio")  >> PC(pool, nameSpace + "beats_loudness_band_ratio");
}
