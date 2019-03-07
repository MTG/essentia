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
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/streaming/algorithms/vectorinput.h>
#include <essentia/scheduler/network.h>
#include "essentia/utils/tnt/tnt2vector.h"
#include "credit_libav.h" 
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;


int main(int argc, char* argv[]) {
  
  if (argc != 3) {
    cout << "Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " <input_sim_matrix_txt> <json_output_path>" << endl;
    creditLibAV();
    exit(1);
  }
  string inputSimFile = argv[1];
  string outputFilename = argv[2];

  /*
  string inputSimFile = "/home/albin/Documents/brahm_coldplay_sim_matrix.txt";
  string outputFilename = "/home/albin/Documents/qmax_brahm_coldplay_stream.json";
  */
 
  vector<vector<Real> > inputSimMatrix;
  // read the 2d array text file and store it to a 2D vector 
  ifstream myReadFile;
  myReadFile.open(inputSimFile);
  string line;
  int i = 0;

  while (getline(myReadFile, line)) {
    Real value;
    stringstream ss(line);
    inputSimMatrix.push_back(vector<Real> ());
    while (ss >> value) {
      inputSimMatrix[i].push_back(value);
    }
    ++i;
  }

  cout << "\nInput sim matrix size: " << inputSimMatrix.size() << ", " << inputSimMatrix[0].size() << "\n" << endl;

  // register the algorithms in the factory
  essentia::init();

  Real disExtension = 0.5;
  Real disOnset = 0.5;

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  // Algorithm* vectorInput = new streaming::VectorInput<TNT::2DArray>();
  Algorithm* vectorInput = new streaming::VectorInput<vector<Real> >(&inputSimMatrix);

  Algorithm* alignment = factory.create("CoverSongSimilarity",
                                  "disExtension", disExtension,
                                  "disOnset", disOnset);


  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos ---------" << endl;

  vectorInput->output("data") >> alignment->input("inputArray");
  alignment->output("scoreMatrix") >> PC(pool, "qmax");

  Network network(vectorInput);
  network.run();

  ///////// Writing results to a json file //////////////
  cout << "-------- writing results to " << outputFilename << " ---------" << endl;
  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", outputFilename,
                                                                   "format", "json");
  output->input("pool").set(pool);
  output->compute();

  delete alignment;
  delete output;

  essentia::shutdown();

  return 0;
}
