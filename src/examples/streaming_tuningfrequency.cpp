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
#include <essentia/essentiamath.h>
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/scheduler/network.h>
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

int main(int argc, char* argv[]) {

  if (argc != 2) {
    cout << "ERROR: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input" << endl;
    exit(1);
  }

  string audioFilename = argv[1];

  // register the algorithms in the factory(ies)
  essentia::init();
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename);

  Algorithm* tf    = factory.create("TuningFrequencyExtractor");

  Pool pool;

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  try {

    audio->output("audio")         >>  tf->input("signal");
    tf->output("tuningFrequency")  >>  PC(pool, "tonal.tuningFrequency");


  /////////// STARTING THE ALGORITHMS //////////////////
    cout << "-------- start processing " << audioFilename << " --------" << endl;

    Network n(audio);
    n.run();

    cout << "Tuning frequency: " 
         << mean(pool.value<std::vector<Real> >("tonal.tuningFrequency")) << endl;
  }
  catch (EssentiaException& e) {
      cout << "EXC: " << e.what() << endl;
  }

  essentia::shutdown();

  return 0;
}
