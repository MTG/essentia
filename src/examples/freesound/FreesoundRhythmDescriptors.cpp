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


#include "FreesoundRhythmDescriptors.h"
using namespace std;
using namespace essentia;
using namespace streaming;


const string FreesoundRhythmDescriptors::nameSpace="rhythm."; 


void  FreesoundRhythmDescriptors::createNetwork(SourceBase& source, Pool& pool){
  
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  // Rhythm extractor
  Algorithm* rhythmExtractor = factory.create("RhythmExtractor2013");
  rhythmExtractor->configure("method","degara");
  
  
  connect(source, rhythmExtractor->input("signal"));
  connect(rhythmExtractor->output("ticks"),        pool, nameSpace + "beats_position");
  connect(rhythmExtractor->output("bpm"),          pool, nameSpace + "bpm");
  connect(rhythmExtractor->output("estimates"),    pool, nameSpace + "bpm_estimates");
  connect(rhythmExtractor->output("bpmIntervals"), pool, nameSpace + "bpm_intervals");
  connect(rhythmExtractor->output("confidence"), NOWHERE);

  // BPM Histogram descriptors
  Algorithm* bpmhist = factory.create("BpmHistogramDescriptors");
  connect(rhythmExtractor->output("bpmIntervals"), bpmhist->input("bpmIntervals"));
  connectSingleValue(bpmhist->output("firstPeakBPM"),     pool, nameSpace + "first_peak_bpm");
  connectSingleValue(bpmhist->output("firstPeakWeight"),  pool, nameSpace + "first_peak_weight");
  connectSingleValue(bpmhist->output("firstPeakSpread"),  pool, nameSpace + "first_peak_spread");
  connectSingleValue(bpmhist->output("secondPeakBPM"),    pool, nameSpace + "second_peak_bpm");
  connectSingleValue(bpmhist->output("secondPeakWeight"), pool, nameSpace + "second_peak_weight");
  connectSingleValue(bpmhist->output("secondPeakSpread"), pool, nameSpace + "second_peak_spread");

      // Onset Detection
  Algorithm* onset = factory.create("OnsetRate");
  connect(source, onset->input("signal"));
  connect(onset->output("onsetTimes"), pool, nameSpace + "onset_times");
  connect(onset->output("onsetRate"), pool, nameSpace + "onset_rate"); 

}

void FreesoundRhythmDescriptors::createBeatsLoudnessNetwork(SourceBase& source, Pool& pool){
  string nameSpace = "rhythm.";
  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();
  Real analysisSampleRate = 44100; // TODO: unify analysisSampleRate
  vector<Real> ticks = pool.value<vector<Real> >(nameSpace + "beats_position");
  Algorithm* beatsLoudness = factory.create("BeatsLoudness",
    "sampleRate", analysisSampleRate,
    "beats", ticks
  );
  connect(source, beatsLoudness->input("signal"));
  connect(beatsLoudness->output("loudness"), pool, nameSpace + "beats_loudness");
  connect(beatsLoudness->output("loudnessBandRatio"), pool, nameSpace + "beats_loudness_band_ratio");
}