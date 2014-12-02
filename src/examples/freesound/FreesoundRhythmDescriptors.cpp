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
  
  
  source >> rhythmExtractor->input("signal");
  rhythmExtractor->output("ticks") >>        PC(pool, nameSpace + "beats_position");
  rhythmExtractor->output("bpm") >>          PC(pool, nameSpace + "bpm");
  rhythmExtractor->output("estimates") >>    NOWHERE;
  rhythmExtractor->output("bpmIntervals") >> PC(pool, nameSpace + "bpm_intervals");
  rhythmExtractor->output("confidence") >>   PC(pool, nameSpace + "bpm_confidence");

  // BPM Histogram descriptors
  Algorithm* bpmhist = factory.create("BpmHistogramDescriptors");
  rhythmExtractor->output("bpmIntervals") >> bpmhist->input("bpmIntervals");
  bpmhist->output("firstPeakBPM") >>     PC(pool, nameSpace + "first_peak_bpm");
  bpmhist->output("firstPeakWeight") >>  PC(pool, nameSpace + "first_peak_weight");
  bpmhist->output("firstPeakSpread") >>  PC(pool, nameSpace + "first_peak_spread");
  bpmhist->output("secondPeakBPM") >>    PC(pool, nameSpace + "second_peak_bpm");
  bpmhist->output("secondPeakWeight") >> PC(pool, nameSpace + "second_peak_weight");
  bpmhist->output("secondPeakSpread") >> PC(pool, nameSpace + "second_peak_spread");

  // Onset Detection
  Algorithm* onset = factory.create("OnsetRate");
  source >> onset->input("signal");
  onset->output("onsetTimes") >> PC(pool, nameSpace + "onset_times");
  onset->output("onsetRate") >> PC(pool, nameSpace + "onset_rate"); 

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
  source >> beatsLoudness->input("signal");
  beatsLoudness->output("loudness") >> PC(pool, nameSpace + "beats_loudness");
  beatsLoudness->output("loudnessBandRatio") >> PC(pool, nameSpace + "beats_loudness_band_ratio");
}