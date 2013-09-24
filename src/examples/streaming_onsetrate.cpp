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
#include <essentia/scheduler/network.h>
#include <essentia/streaming/algorithms/poolstorage.h>
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

int main(int argc, char* argv[]) {
  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  if (argc != 2) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile" << endl;
    exit(1);
  }
  string audioFilename = argv[1];

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audioloader = factory.create("AudioLoader",
                                          "filename", audioFilename);

  Algorithm* mono        = factory.create("MonoMixer");

  Algorithm* onsetrate   = factory.create("OnsetRate");

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  // audioloader -> mono
  audioloader->output("audio")           >>  mono->input("audio");
  audioloader->output("numberChannels")  >>  mono->input("numberChannels");
  audioloader->output("sampleRate")      >>  PC(pool, "metadata.sampleRate");

  // mono -> onsetrate
  mono->output("audio")                  >>  onsetrate->input("signal");

  onsetrate->output("onsetTimes")        >>  PC(pool, "rhythm.onsetTimes");
  onsetrate->output("onsetRate")         >>  PC(pool, "rhythm.onsetRate");

  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  Network network(audioloader);

  network.run();

  // printing results to file
  cout << "-------- results --------" << endl;

  int totalSamples = audioloader->output("audio").totalProduced();
  Real fileLength = totalSamples / pool.value<Real>("metadata.sampleRate");

  const vector<Real>& onsetTimes = pool.value<vector<Real> >("rhythm.onsetTimes");

  cout << "onsetRate: " << onsetTimes.size() / fileLength << endl;
  cout << "onsetTimes: " << onsetTimes << endl;

  essentia::shutdown();

  return 0;
}
