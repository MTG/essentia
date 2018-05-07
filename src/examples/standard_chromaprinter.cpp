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
    cout << "Usage: " << argv[0] << " audio_input output_file chromaprint_duration" << endl;
    creditLibAV();
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];
  Real chromaprintDuration = stof(argv[3]);

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  /////// PARAMS //////////////
  Real sampleRate = 44100.0;

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* audioLoader = factory.create("MonoLoader",
                                          "filename", audioFilename,
                                          "sampleRate", sampleRate);

  Algorithm* chromaPrinter = factory.create("Chromaprinter");

  vector<Real> audio;
  string chromaprint;
  Real confidence;

  audioLoader->output("audio").set(audio);
  audioLoader->compute();

  chromaPrinter->input("signal").set(audio);
  chromaPrinter->output("fingerprint").set(chromaprint);
  chromaPrinter->compute();


  // output to file:
  Algorithm* yamlOutput = factory.create("YamlOutput",
                                         "filename", outputFilename);

  pool.add("pool.chromaprint", chromaprint);

  yamlOutput->input("pool").set(pool);

  // run algorithms
  yamlOutput->compute();

  delete audioLoader;
  delete chromaPrinter;

  essentia::shutdown();

  return 0;
}
