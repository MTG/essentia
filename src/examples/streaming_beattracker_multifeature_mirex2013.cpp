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
#include <fstream> // to write ticks to output file
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

  cout << "Multifeature beat tracker based on BeatTrackerMultiFeature algorithm." << endl;
  cout << "Outputs beat positions in MIREX 2013 format." << endl;

  if (argc != 3) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile" << " output_results_file" << endl;
    exit(1);
  }
  string audioFilename = argv[1];
  string outputFilename = argv[2];

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* monoloader = factory.create("MonoLoader", "filename", audioFilename);
  Algorithm* beattracker = factory.create("BeatTrackerMultiFeature");

  monoloader->configure("sampleRate", 44100.);

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  // monoloader will always resample to 44100
  monoloader->output("audio")       >> beattracker->input("signal");
  beattracker->output("ticks")      >> PC(pool, "rhythm.ticks");
  beattracker->output("confidence") >> NOWHERE;

  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  Network network(monoloader);
  network.run();


  // writing results to file
  vector<Real> ticks;
  if (pool.contains<vector<Real> >("rhythm.ticks")) { // there might be empty ticks
    ticks = pool.value<vector<Real> >("rhythm.ticks");
  }
  ostream* fileStream = new ofstream(outputFilename.c_str());
  for (size_t i=0; i<ticks.size(); ++i) {
    *fileStream << ticks[i] << "\n";
  }
  delete fileStream;

  essentia::shutdown();
  return 0;
}
