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

    results.set("metadata.audio_properties.equal_loudness", false);
    results.set("metadata.version.freesound_extractor", EXTRACTOR_VERSION);
  
    FreesoundLowlevelDescriptors *lowlevel = new FreesoundLowlevelDescriptors();
    FreesoundRhythmDescriptors *rhythm = new FreesoundRhythmDescriptors();
    FreesoundTonalDescriptors *tonal = new FreesoundTonalDescriptors();
    FreesoundSfxDescriptors *sfx = new FreesoundSfxDescriptors();
    
    Algorithm* loader = AlgorithmFactory::create("EasyLoader",
                                                 "filename",   audioFilename,
                                                 "sampleRate", SAMPLE_RATE);
    
    lowlevel->createNetwork(loader->output("audio"),results);
    rhythm->createNetwork(loader->output("audio"),results);
    tonal->createNetwork(loader->output("audio"),results);
    sfx->createNetwork(loader->output("audio"),results);

    Network network(loader,false);
    network.run();

    // Descriptors that require values from other descriptors in the previous chain
    vector<Real> pitch = results.value<vector<Real> >("lowlevel.pitch");
    VectorInput<Real> *pitchVector = new VectorInput<Real>();
    pitchVector->setVector(&pitch);

    delete loader;
    loader = AlgorithmFactory::create("EasyLoader",
                            "filename",   audioFilename,
                            "sampleRate", SAMPLE_RATE);
    
    rhythm->createBeatsLoudnessNetwork(loader->output("audio"), results);
    sfx->createHarmonicityNetwork(loader->output("audio"), results);

    Network rhythmAndSfxNetwork(loader,false);
    rhythmAndSfxNetwork.run();

    sfx->createPitchNetwork(*pitchVector, results);
    Network sfxPitchNetwork(pitchVector);
    sfxPitchNetwork.run();

    lowlevel->computeAverageLoudness(results);

    cout << "Compute Aggregation"<<endl;
    
    this->stats = this->computeAggregation(results);
    
    cout << "All done"<<endl;
    
    delete loader;
    
    return;
 }


Pool FreesoundExtractor::computeAggregation(Pool& pool){

    // choose which descriptors stats to output
    const char* defaultStats[] = {
        "mean", "var", "min", "max","dmean", "dmean2", "dvar", "dvar2", "median"
    };
 
    map<string, vector<string> > exceptions;
    
    const char *noStatsSfxArray[] = {
        "der_av_after_max", "effective_duration","flatness", "logattacktime",
        "max_der_before_max", "oddtoevenharmonicenergyratio", "pitch_centroid",
        "temporal_centroid","temporal_decrease" ,"temporal_kurtosis",
        "temporal_skewness","temporal_spread"};
    
    vector<string> noStatsSfx = arrayToVector<string>(noStatsSfxArray);
    
    for (int i=0; i<(int)noStatsSfx.size(); i++) {
        exceptions["sfx."+noStatsSfx[i]] = arrayToVector<string>(defaultStats);
    }

    standard::Algorithm* aggregator =
        standard::AlgorithmFactory::create("PoolAggregator",
                                           "defaultStats",
                                           arrayToVector<string>(defaultStats),
                                           "exceptions", exceptions);
    Pool poolStats;
    aggregator->input("input").set(pool);
    aggregator->output("output").set(poolStats);
    aggregator->compute();

    
    // variable descriptor length counts
    poolStats.set(string("rhythm.onset_count"),
                  pool.value<vector<Real> >("rhythm.onset_times").size());
    poolStats.set(string("rhythm.beats_count"),
                  pool.value<vector<Real> >("rhythm.beats_position").size());
    poolStats.set(string("tonal.chords_count"),
                  pool.value<vector<string> >("tonal.chords_progression").size());
    
    // hpcp_mean peak count
    vector<Real> hpcp_peak_amps, hpcp_peak_pos;
    standard::Algorithm* hpcp_peaks =
        standard::AlgorithmFactory::create("PeakDetection", "threshold",0.1);

    hpcp_peaks->input("array")
        .set(poolStats.value<vector<Real> >("tonal.hpcp.mean"));
    hpcp_peaks->output("amplitudes")
        .set(hpcp_peak_amps);
    hpcp_peaks->output("positions")
        .set(hpcp_peak_pos);
    hpcp_peaks->compute();
    
    poolStats.set(string("tonal.hpcp_peak_count"), hpcp_peak_amps.size());
    
    delete aggregator;
    delete hpcp_peaks;
    
    return poolStats;
}


void FreesoundExtractor::outputToFile(Pool& pool, const string& outputFilename,
                                      bool outputJSON){

  cout << "Writing results to file " << outputFilename << endl;

  standard::Algorithm* output =
    standard::AlgorithmFactory::create("YamlOutput",
                                       "filename", outputFilename,
                                       "doubleCheck", true,
                                       "format", outputJSON ? "json" : "yaml");
  output->input("pool").set(pool);
  output->compute();
  delete output;
}
