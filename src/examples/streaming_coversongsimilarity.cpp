/*
 * Copyright (C) 2006-2020 Music Technology Group - Universitat Pompeu Fabra
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
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/streaming/algorithms/vectorinput.h>
#include <essentia/scheduler/network.h>
#include "credit_libav.h" 

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;


int main(int argc, char* argv[]) {
  
  if (argc != 4) {
    cout << "Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << "<query_audio_file> <reference_feature_text_file> <json_output_path>" << endl;
    creditLibAV();
    exit(1);
  }
  string queryAudioFile = argv[1]; 
  string referenceFile = argv[2];
  string outputFilename = argv[3];
  vector<vector<Real> > referenceFeature;

  // read the 2d array text file and store it to a 2D vector 
  ifstream myReadFile;
  myReadFile.open(referenceFile);
  string line;
  int i = 0;

  while (getline(myReadFile, line)) {
    Real value;
    stringstream ss(line);
    referenceFeature.push_back(vector<Real> ());
    while (ss >> value) {
      referenceFeature[i].push_back(value);
    }
    ++i;
  }

  cout << "Input sim matrix size: " << referenceFeature.size() << ", " << referenceFeature[0].size() << "\n" << endl;
  // register the algorithms in the factory
  essentia::init();
  Pool pool;
  Real disExtension = 0.5;
  Real disOnset = 0.5;

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  // Algorithm* vectorInput = new streaming::VectorInput<vector<Real> >(&inputSimMatrix);

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", queryAudioFile,
                                    "sampleRate", 44100);
  

  Algorithm* fc    = factory.create("FrameCutter",
                                    "frameSize", 4096,
                                    "hopSize", 2048);

  Algorithm* w     = factory.create("Windowing",
                                    "type", "blackmanharris62");

  Algorithm* spec  = factory.create("Spectrum");

  Algorithm* peak = factory.create("SpectralPeaks",
                                   "sampleRate", 44100);

  Algorithm* white = factory.create("SpectralWhitening",
                                    "maxFrequency", 3500,
                                    "sampleRate", 44100);

  Algorithm* hpcp = factory.create("HPCP",
                                   "sampleRate", 44100,
                                   "minFrequency", 100,
                                   "maxFrequency", 3500,
                                   "size", 12);

  // with default params
  Algorithm* csm = factory.create("ChromaCrossSimilarity",
                                  "referenceFeature", referenceFeature);

  Algorithm* alignment = factory.create("CoverSongSimilarity",
                                  "disExtension", disExtension,
                                  "disOnset", disOnset,
                                  "distanceType", "asymmetric",
                                  "pipeDistance", true);

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos ---------" << endl;

  audio->output("audio") >> fc->input("signal");
  fc->output("frame") >> w->input("frame");
  w->output("frame") >> spec->input("frame");
  spec->output("spectrum") >> peak->input("spectrum");
  spec->output("spectrum") >> white->input("spectrum");
  peak->output("magnitudes") >> white->input("magnitudes");
  peak->output("frequencies") >> white->input("frequencies");
  peak->output("frequencies") >> hpcp->input("frequencies"); 
  white->output("magnitudes") >> hpcp->input("magnitudes");
  hpcp->output("hpcp") >> csm->input("queryFeature");
  csm->output("csm") >> alignment->input("inputArray");
  alignment->output("scoreMatrix") >> PC(pool, "scoreMatrix");
  alignment->output("distance") >> PC(pool, "distance");

  Network network(audio);
  network.run();

  ///////// Writing results to a json file //////////////
  cout << "-------- writing results to " << outputFilename << " ---------" << endl;
  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", outputFilename,
                                                                   "format", "json");
  output->input("pool").set(pool);
  output->compute();
  cout << "---- Done ----" << endl;

  delete output;

  essentia::shutdown();

  return 0;
}

