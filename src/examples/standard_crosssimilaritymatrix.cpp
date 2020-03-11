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
#include <essentia/essentiamath.h>
#include <essentia/pool.h>
#include "credit_libav.h"
#include "essentia/utils/tnt/tnt2vector.h"

using namespace std;
using namespace essentia;
using namespace essentia::standard;

vector<vector<Real> > readMatrixFile(string inputFileName);

int main(int argc, char* argv[]) {
  
  if (argc != 4) {
    cout << "Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " <query_feature_text_file> <reference_feature_text_file> <json_output_path>" << endl;
    creditLibAV();
    exit(1);
  }

  // register the algorithms in the factory(ies)
  essentia::init();
  // input feature matrix as text file
  string queryFilename = argv[1];
  string referenceFilename = argv[2];
  string outputFilename = argv[3];

  vector<vector<Real> > queryFeature = readMatrixFile(queryFilename);
  vector<vector<Real> > referenceFeature = readMatrixFile(referenceFilename);

  cout << "------------ Inputs -------------" << endl;
  cout << "Query shape: " << queryFeature.size() << ", " << queryFeature[0].size() << endl;
  cout << "Reference shape: " << referenceFeature.size() << ", " << referenceFeature[0].size() << endl;

  /////// PARAMS //////////////
  Real binarizePercentile = 0.095;
  int frameStackStride = 1;
  int frameStackSize = 9;
  bool binarize = false;

  Pool pool;
  AlgorithmFactory& factory = standard::AlgorithmFactory::instance();
  Algorithm* csm = factory.create("CrossSimilarityMatrix",
                                  "binarize", binarize,
                                  "binarizePercentile", binarizePercentile,
                                  "frameStackSize", frameStackSize,
                                  "frameStackStride", frameStackStride);

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos ---------" << endl;
  vector<vector<Real> > csmout;
  csm->input("queryFeature").set(queryFeature);
  csm->input("referenceFeature").set(referenceFeature);
  csm->output("csm").set(csmout);
  
  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-----Cross-similarity matrix calculation --------" << endl;
  csm->compute();
  // since pool only supports TNT::Array2D
  TNT::Array2D<Real> outArray = vecvecToArray2D(csmout);
  // add to pool
  pool.add("csm", outArray);
  cout << "Output matrix size: " << csmout.size() << "\t" << csmout[0].size() << endl;

  ///////// Writing results to a json file //////////////
  cout << "-------- writing results to json file '" << outputFilename << "' ---------" << endl;
  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", outputFilename,
                                                                   "format", "json");
  output->input("pool").set(pool);
  output->compute();
  cout << "------------ Done --------------" << endl;

  delete csm;
  essentia::shutdown();
  return 0;

}

// read the 2d array text file and store it to a 2D vector 
vector<vector<Real> > readMatrixFile(string inputFileName) {
  ifstream myReadFile;
  myReadFile.open(inputFileName);
  std::string line;
  int i = 0;
  vector<vector<Real> > outputArray;

  while (getline(myReadFile, line)) {
    Real value;
    stringstream ss(line);
    outputArray.push_back(vector<Real> ());
    while (ss >> value) {
      outputArray[i].push_back(value);
    }
    ++i;
  }
  return outputArray;
}