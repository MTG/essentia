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
#include <essentia/algorithmfactory.h>
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/scheduler/network.h>
#include "credit_libav.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;


bool hasFlag(char** begin, char** end, const string& option) {
  return find(begin, end, option) != end;
}

string getArgument(char** begin, char** end, const string& option) {
  char** iter = find(begin, end, option);
  if (iter != end && ++iter != end) return *iter;

  return string();
}

void printHelp(string fileName) {
    cout << "Usage: " << fileName << " pb_graph audio_input output_json [--help|-h] [--list-nodes|-l] [--patchwise|-p] [[-output-node|-o] node_name]" << endl;
    cout << "  -h, --help: print this help" << endl;
    cout << "  -l, --list-nodes: list the nodes in the input graph (model)" << endl;
    cout << "  -p, --patchwise: write out patch-wise predctions (one per patch) instead of averaging them" << endl;
    cout << "  -o, --output-node: node (layer) name to retrieve from the graph (default: model/Sigmoid)" << endl;
    creditLibAV();
}

vector<string> flags({"-h", "--help",
                      "-l", "--list-nodes",
                      "-p", "--patchwise",
                      "-o", "--output-node"});


int main(int argc, char* argv[]) {
  // Sanity check for the command line options.
  for (char** iter = argv; iter < argv + argc; ++iter) {
    if (**iter == '-') {
      string flag(*iter);
      if (find(flags.begin(), flags.end(), flag) == flags.end()){
        cout << argv[0] << ": invalid option '" << flag << "'" << endl;
        printHelp(argv[0]);
        exit(1);
      }
    }
  }

  if (hasFlag(argv, argv + argc, "--help") ||
      hasFlag(argv, argv + argc, "-h")) {
    printHelp(argv[0]);
    exit(0);
  }

  string outputLayer = "model/Sigmoid";

  if (hasFlag(argv, argv + argc, "--list-nodes") ||
      hasFlag(argv, argv + argc, "-l")) {
    outputLayer = "";

  } else if (hasFlag(argv, argv + argc, "--output-node") ) {
    outputLayer = getArgument(argv, argv + argc, "--output-node");
  
  } else if (hasFlag(argv, argv + argc, "-o") ) {
    outputLayer = getArgument(argv, argv + argc, "-o");
  }

  if ((argc < 4) || (argc > 8)) {
    cout << argv[0] <<": incorrect number of arguments." << endl;
    printHelp(argv[0]);
    exit(1);
  }

  string graphName = argv[1];
  string audioFilename = argv[2];
  string outputFilename = argv[3];

  // rather to output the patch-wise predictions or to average them.
  const bool average = (hasFlag(argv, argv + argc, "--patchwise") ||
                        hasFlag(argv, argv + argc, "-p")) ? false : true;

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;
  Pool aggrPool;  // a pool for the the aggregated predictions
  Pool* poolPtr = &pool;

  /////// PARAMS //////////////
  Real sampleRate = 16000.0;

  AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate);

  Algorithm* tfp   = factory.create("TensorflowPredictMusiCNN",
                                    "graphFilename", graphName,
                                    "output", outputLayer);

  // If the output layer is empty, we have already printed the list of nodes.
  // Exit now.
  if (outputLayer.empty()){
    essentia::shutdown();

    return 0;
  }

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  audio->output("audio")     >>  tfp->input("signal");
  tfp->output("predictions") >>  PC(pool, "predictions");


  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  // create a network with our algorithms...
  Network n(audio);
  // ...and run it, easy as that!
  n.run();

  if (average) {
    // aggregate the results
    cout << "-------- averaging the predictions --------" << endl;

    const char* stats[] = {"mean"};

    standard::Algorithm* aggr = standard::AlgorithmFactory::create("PoolAggregator",
                                                                  "defaultStats", arrayToVector<string>(stats));

    aggr->input("input").set(pool);
    aggr->output("output").set(aggrPool);
    aggr->compute();

    poolPtr = &aggrPool;

    delete aggr;
  }

  // write results to file
  cout << "-------- writing results to json file " << outputFilename << " --------" << endl;

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "format", "json",
                                                                   "filename", outputFilename);
  output->input("pool").set(*poolPtr);
  output->compute();
  n.clear();

  delete output;
  essentia::shutdown();

  return 0;
}
