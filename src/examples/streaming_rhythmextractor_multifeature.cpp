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

#include <iostream>
#include <essentia/algorithmfactory.h>
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/scheduler/network.h>
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

int main(int argc, char* argv[]) {
  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  cout << "Rhythm extractor (beat tracker, BPM, positions of tempo changes) based on multifeature beat tracker (see the BeatTrackerMultiFeature algorithm)" << endl;

  if (argc != 2) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile" << endl;
    exit(1);
  }
  string audioFilename = argv[1];

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* monoloader = factory.create("MonoLoader",
                                         "filename", audioFilename,
                                         "sampleRate", 44100.);
 // using 'multifeature' method for best accuracy, 
 // but it requires the largest computation time
  Algorithm* rhythmextractor = factory.create("RhythmExtractor2013",
                                              "method", "multifeature");


  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  monoloader->output("audio")             >> rhythmextractor->input("signal");
  rhythmextractor->output("ticks")        >> PC(pool, "rhythm.ticks");
  rhythmextractor->output("confidence")   >> PC(pool, "rhythm.ticks_confidence");
  rhythmextractor->output("bpm")          >> PC(pool, "rhythm.bpm");
  rhythmextractor->output("estimates")    >> PC(pool, "rhythm.estimates");
  rhythmextractor->output("bpmIntervals") >> PC(pool, "rhythm.bpmIntervals");
  // FIXME we need better rubato estimation algorithm
  //connect(rhythmextractor->output("rubatoStart"), pool, "rhythm.rubatoStart");
  //connect(rhythmextractor->output("rubatoStop"), pool, "rhythm.rubatoStop");
  //connect(rhythmextractor->output("rubatoNumber"), pool, "rhythm.rubatoNumber");


  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  Network network(monoloader);
  network.run();


  // printing results
  cout << "-------- results --------" << endl;
  cout << "bpm: " << pool.value<Real>("rhythm.bpm") << endl;
  cout << "ticks: " << pool.value<vector<Real> >("rhythm.ticks") << endl;
  cout << "ticks detection confidence: " << pool.value<Real>("rhythm.ticks_confidence") << endl; 
  cout << "estimates: " << pool.value<vector<Real> >("rhythm.estimates") << endl;
  cout << "bpmIntervals: " << pool.value<vector<Real> >("rhythm.bpmIntervals") << endl;
  //cout << "rubatoNumber:" << (int) pool.value<Real>("rhythm.rubatoNumber") << endl;
  //try {
  //    cout << "rubatoStart: " << pool.value<vector<Real> >("rhythm.rubatoStart") << endl;
  //    cout << "rubatoStop: " << pool.value<vector<Real> >("rhythm.rubatoStop") << endl;
  //}
  //catch (EssentiaException&) {
  //  cout << "No rubato regions found" << endl;
  //}

  essentia::shutdown();

  return 0;
}
