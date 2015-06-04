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
using namespace scheduler;

int main(int argc, char* argv[]) {

  if (argc < 3 ) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile output_yamlfile [1/0 print to stdout]" << endl;
    exit(1);
  }

  // register the algorithms in the factory(ies)
  essentia::init();

  /////// PARAMS //////////////
  // don't change these default values as they guarantee that pitch extractor output
  // is correct, no tests were done on other values
  int framesize = 2048;
  int hopsize = 128;
  int sr = 44100;


  // instantiate factory and create algorithms:
  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audioload = factory.create("MonoLoader",
                                        "filename", argv[1],
                                        "sampleRate", sr,
                                        "downmix", "mix");

  Algorithm* equalLoudness = factory.create("EqualLoudness");
  Algorithm* predominantMelody = factory.create("PredominantMelody",
                                                "frameSize", framesize,
                                                "hopSize", hopsize,
                                                "sampleRate", sr);
  // data storage
  Pool pool;

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  // audio -> equal loudness -> predominant melody
  audioload->output("audio")                   >> equalLoudness->input("signal");
  equalLoudness->output("signal")              >> predominantMelody->input("signal");
  predominantMelody->output("pitch")           >> PC(pool, "tonal.predominant_melody.pitch");
  predominantMelody->output("pitchConfidence") >> PC(pool, "tonal.predominant_melody.pitch_confidence");



  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << argv[1]<< " --------" << endl;

  Network network(audioload);
  network.run();

  // write results to yamlfile
  cout << "-------- writing results to file " << argv[2] << " --------" << endl;

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", argv[2]);
  output->input("pool").set(pool);
  output->compute();

  if (argc == 4 && atoi(argv[3])) {
    // printing to stdout:
    cout << "number of frames: " << pool.value<vector<Real> >("tonal.predominant_melody.pitch").size() << endl;
  }

  // clean up:
  delete output;
  essentia::shutdown();

  return 0;
}
