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
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/streaming/algorithms/vectorinput.h>
#include <essentia/scheduler/network.h>
#include "credit_libav.h" 
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

vector<vector<Real> > readMatrixFile(string inputFileName);

int main(int argc, char* argv[]) {
  
  if (argc != 4) {
    cout << "Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " <query_song_path> <reference_feature_file_path> <json_output_path>" << endl;
    creditLibAV();
    exit(1);
  }

  string queryFilename = argv[1];
  string referenceFilename = argv[2];
  string outputFilename = argv[3];
  vector<vector<Real> > referenceFeature = readMatrixFile(referenceFilename);

  cout << "----------- Inputs -----------" << endl;
  cout << "Reference song input shape: " << referenceFeature.size() << ", " << referenceFeature[0].size() << endl;

  // register the algorithms in the factory
  essentia::init();
  Pool pool;

  /////// PARAMS //////////////
  Real sampleRate = 44100.0;
  int frameSize = 4096;
  int hopSize = 2048;
  int numBins = 12;
  Real minFrequency = 100;
  Real maxFrequency = 3500;
  int oti = 3; // hardcoded, should be obtained from the oti algo
  bool otiBinary = false;
  Real binarizePercentile = 0.095;
  int frameStackStride = 1;
  int frameStackSize = 9;

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  // we want to compute the HPCP of a file: we need the create the following:
  // audioloader -> framecutter -> windowing -> Spectrum -> SpectralPeaks -> SpectralWhitening -> HPCP
  // Later we want compute cross similarity between these input HPCP features
  // HPCP -> CrossSimilarityMatrix 

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", queryFilename,
                                    "sampleRate", sampleRate);
  

  Algorithm* fc    = factory.create("FrameCutter",
                                    "frameSize", frameSize,
                                    "hopSize", hopSize);

  Algorithm* w     = factory.create("Windowing",
                                    "type", "blackmanharris62");

  Algorithm* spec  = factory.create("Spectrum");

  Algorithm* peak = factory.create("SpectralPeaks",
                                   "sampleRate", sampleRate);

  Algorithm* white = factory.create("SpectralWhitening",
                                    "maxFrequency", maxFrequency,
                                    "sampleRate", sampleRate);

  Algorithm* hpcp = factory.create("HPCP",
                                   "sampleRate", sampleRate,
                                   "minFrequency", minFrequency,
                                   "maxFrequency", maxFrequency,
                                   "size", numBins);

  Algorithm* csm = factory.create("ChromaCrossSimilarity",
                                  "referenceFeature", referenceFeature,
                                  "otiBinary", otiBinary,
                                  "oti", oti,
                                  "binarizePercentile", binarizePercentile,
                                  "frameStackSize", frameStackSize,
                                  "frameStackStride", frameStackStride);

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos for hpcp and csm extraction ---------" << endl;

 // Audio -> FrameCutter
  audio->output("audio")    >>  fc->input("signal");

  // FrameCutter -> Windowing -> Spectrum
  fc->output("frame")       >>  w->input("frame");
  w->output("frame")        >>  spec->input("frame");

  // Spectrum -> Spectral Peaks -> Spectral Whitening
  spec->output("spectrum")    >>  peak->input("spectrum");
  peak->output("frequencies") >>  white->input("frequencies");
  spec->output("spectrum")    >>  white->input("spectrum");
  peak->output("magnitudes")  >>  white->input("magnitudes");

  // Spectral whitening -> HPCP -> Pool
  peak->output("frequencies") >>  hpcp->input("frequencies");
  white->output("magnitudes") >>  hpcp->input("magnitudes");

  // HPCP -> CSM -> pool
  hpcp->output("hpcp") >> csm->input("queryFeature");
  csm->output("csm") >> PC(pool, "csm");
  
  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << queryFilename << " --------" << endl;

  Network network(audio);
  network.run();

  ///////// Writing results to a json file //////////////
  cout << "-------- writing results to " << outputFilename << " ---------" << endl;
  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", outputFilename,
                                                                   "format", "json");
  output->input("pool").set(pool);
  output->compute();
  cout << "...Done..." << endl;
  
  delete audio;
  delete fc;
  delete w;
  delete spec;
  delete peak;
  delete white;
  delete hpcp;
  delete csm;
  delete output;

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


