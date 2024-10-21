/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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
#include <fstream>
#include <essentia/algorithmfactory.h>
#include <essentia/pool.h>
#include "credit_libav.h"
using namespace std;
using namespace essentia;
using namespace standard;

int main(int argc, char* argv[]) {

  if (argc != 4) {
    cout << "Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input output_file graph_file" << endl;
    creditLibAV();
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];

  // define graphFilePath
  string graphFilePath = argv[3];

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  /////// PARAMS //////////////
  Real sampleRate = 11025.0;
  int resampleQuality = 4;

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* audioLoader = factory.create("MonoLoader",
                                          "filename", audioFilename,
                                          "sampleRate", sampleRate,
                                          "resampleQuality", resampleQuality);

  Algorithm* tempoCNN = factory.create("TempoCNN",
                                        "graphFilename", graphFilePath);

  // inputs and outputs
  vector<Real> audio;
  Real globalTempo;
  vector<Real> localTempo;
  vector<Real> localTempoProbabilities;

  // process
  audioLoader->output("audio").set(audio);
  audioLoader->compute();

  tempoCNN->input("audio").set(audio);
  tempoCNN->output("globalTempo").set(globalTempo);
  tempoCNN->output("localTempo").set(localTempo);
  tempoCNN->output("localTempoProbabilities").set(localTempoProbabilities);
  tempoCNN->compute();

  pool.add("tempoCNN.global_tempo", globalTempo);
  pool.add("tempoCNN.localTempo", localTempo);
  pool.add("tempoCNN.localTempoProbabilities", localTempoProbabilities);

  // output results
  cout << "------------- writing results to file " << outputFilename << " -------------" << endl;

  Algorithm* json = factory.create("YamlOutput",
                                    "filename", outputFilename,
                                    "format", "json");
  json->input("pool").set(pool);
  json->compute();

  // cleanup
  delete audioLoader;
  delete tempoCNN;
  delete json;

  essentia::shutdown();

  return 0;
}
