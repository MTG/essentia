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

#include "FreesoundExtractor.h"
using namespace std;
using namespace essentia;
using namespace streaming;
using namespace scheduler;

void FreesoundExtractor::compute(const string& audioFilename){

   streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();
   Real analysisSampleRate = 44100;
   results.set("metadata.audio_properties.equal_loudness", false); 
   results.set("metadata.version.freesound_extractor", EXTRACTOR_VERSION); 

   Algorithm* loader = factory.create("EasyLoader",
                                      "filename",   audioFilename,
                                      "sampleRate", analysisSampleRate);
  
   SourceBase& source = loader->output("audio");

  
  FreesoundLowlevelDescriptors *lowlevel = new FreesoundLowlevelDescriptors();
  FreesoundRhythmDescriptors *rhythm = new FreesoundRhythmDescriptors();
  FreesoundTonalDescriptors *tonal = new FreesoundTonalDescriptors();
  FreesoundSfxDescriptors *sfx = new FreesoundSfxDescriptors();

  lowlevel->createNetwork(source,results);
  rhythm->createNetwork(source,results);
  tonal->createNetwork(source,results);
  sfx->createNetwork(source,results);

  Network network(loader,false);
  network.run();

  // Descriptors that require values from other descriptors in the previous chain

  vector<Real> pitch = results.value<vector<Real> >("lowlevel.pitch");
  VectorInput<Real> *pitchVector = new VectorInput<Real>();
  pitchVector->setVector(&pitch);

  Algorithm* loader2 = factory.create("EasyLoader",
                                      "filename",   audioFilename,
                                      "sampleRate", analysisSampleRate);
  rhythm->createBeatsLoudnessNetwork(loader2->output("audio"), results);
  sfx->createHarmonicityNetwork(loader2->output("audio"), results); 

  Network network2(loader2,false);
  network2.run();

  sfx->createPitchNetwork(*pitchVector, results);

  Network sfxPitchNetwork(pitchVector);
  sfxPitchNetwork.run();

  lowlevel->computeAverageLoudness(results);

  cout << "Compute Aggregation"<<endl; 
  this->stats = this->computeAggregation(results);

  cout << "All done"<<endl;
  return;
 }

Pool FreesoundExtractor::computeAggregation(Pool& pool){

  // choose which descriptors stats to output
  const char* defaultStats[] = { "mean", "var", "min", "max", "dmean", "dmean2", "dvar", "dvar2" };
 
  map<string, vector<string> > exceptions;
  //TODO: review exceptions

  standard::Algorithm* aggregator = standard::AlgorithmFactory::create("PoolAggregator",
                                                                       "defaultStats", arrayToVector<string>(defaultStats),
                                                                       "exceptions", exceptions);
  Pool poolStats;
  aggregator->input("input").set(pool);
  aggregator->output("output").set(poolStats);

  aggregator->compute();

  // add descriptors that may be missing due to content
  const Real emptyVector[] = { 0, 0, 0, 0, 0, 0};
  
  int statsSize = int(sizeof(defaultStats)/sizeof(defaultStats[0]));

  if(!pool.contains<vector<Real> >("rhythm.beats_loudness")){
    for (uint i=0; i<statsSize; i++)
        poolStats.set(string("rhythm.beats_loudness.")+defaultStats[i],0); 
    }
  if(!pool.contains<vector<vector<Real> > >("rhythm.beats_loudness_band_ratio"))
    for (uint i=0; i<statsSize; i++) 
      poolStats.set(string("rhythm.beats_loudness_band_ratio.")+defaultStats[i],
        arrayToVector<Real>(emptyVector));
  else if (pool.value<vector<vector<Real> > >("rhythm.beats_loudness_band_ratio").size()<2){
      poolStats.remove(string("rhythm.beats_loudness_band_ratio"));
      for (uint i=0; i<statsSize; i++) {
        if(i==1 || i==6 || i==7)// var, dvar and dvar2 are 0
          poolStats.set(string("rhythm.beats_loudness_band_ratio.")+defaultStats[i],
              arrayToVector<Real>(emptyVector));
        else
          poolStats.set(string("rhythm.beats_loudness_band_ratio.")+defaultStats[i],
              pool.value<vector<vector<Real> > >("rhythm.beats_loudness_band_ratio")[0]);
      }
  }

  delete aggregator;

  return poolStats;
}

void FreesoundExtractor::outputToFile(Pool& pool, const string& outputFilename, bool outputJSON){

  cout << "Writing results to file " << outputFilename << endl;

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", outputFilename,
                                                                   "doubleCheck", true,
                                                                   "format", outputJSON ? "json" : "yaml");
  output->input("pool").set(pool);
  output->compute();
  delete output;
}
