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

#include <iostream>
#include <essentia/algorithmfactory.h>
#include <essentia/scheduler/network.h>
#include <essentia/streaming/algorithms/poolstorage.h>

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace scheduler;


int main(int argc, char* argv[]) {
  if (argc < 3 ) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile output_yamlfile [1/0 print to stdout]" << endl;
    exit(1);
  }

  // Register the algorithms in the factory(ies).
  essentia::init();

  /////// PARAMS //////////////
  // Don't change these default values as they guarantee that pitch extractor output
  // is correct, no tests were done on other values.
  int sr = 44100;

  // Instantiate factory and create algorithms:
  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audioload = factory.create("MonoLoader",
                                        "filename", argv[1],
                                        "sampleRate", sr,
                                        "downmix", "mix");

  Algorithm* humDetector = factory.create("HumDetector");

  Pool pool;

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  // audio -> equal loudness -> predominant melody
  audioload->output("audio")                   >> humDetector->input("signal");
  humDetector->output("frequencies")           >> PC(pool, "frequencies");
  humDetector->output("saliences")             >> PC(pool, "saliences");
  humDetector->output("starts")                >> PC(pool, "starts");
  humDetector->output("ends")                  >> PC(pool, "ends");
  humDetector->output("r")                     >> PC(pool, "r");

  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << argv[1]<< " --------" << endl;

  Network network(audioload);
  network.run();

  // Write results to yamlfile.
  cout << "-------- writing results to file " << argv[2] << " --------" << endl;

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", argv[2]);
  output->input("pool").set(pool);
  output->compute();

  delete output;
  essentia::shutdown();

  return 0;
}
