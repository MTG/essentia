/*
 * Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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

int main(int argc, char* argv[]) {

  if (argc != 4) {
    cout << "Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input model output" << endl;
    creditLibAV();
    exit(1);
  }

  string audioFilename = argv[1];
  string modelName = argv[2];
  string outputFilename = argv[3];

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  /////// PARAMS //////////////
  Real sampleRate = 44100.0;
  int frameSize = 1024;
  int hopSize = 1024;
  vector<int> inputShape({-1, 1, 128, 128});
  vector<string> inputs({"model/Placeholder"});
  vector<string> outputs({"model/Softmax"});

  // we want to compute the MFCC of a file: we need the create the following:
  // audioloader -> framecutter -> windowing -> FFT -> MFCC -> PoolStorage
  // we also need a DevNull which is able to gobble data without doing anything
  // with it (required otherwise a buffer would be filled and blocking)

  AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate);

  Algorithm* fc    = factory.create("FrameCutter",
                                    "frameSize", frameSize,
                                    "hopSize", hopSize,
                                    "startFromZero", true);

  Algorithm* w     = factory.create("Windowing",
                                    "type", "hann",
                                    "zeroPadding", frameSize);

  Algorithm* spec  = factory.create("Spectrum",
                                    "size", frameSize);

  Algorithm* mel  = factory.create("MelBands",
                                   "numberBands", 128,
                                   "type", "magnitude");

  Algorithm* logNorm  = factory.create("UnaryOperator",
                                       "type", "log");
                                   
  Algorithm* vtt      = factory.create("VectorRealToTensor",
                                       "shape", inputShape,
                                       "lastPatchMode", "repeat",
                                       "patchHopSize", 1);

  Algorithm* ttp      = factory.create("TensorToPool",
                                       "mode", "overwrite",
                                       "namespace", "model/Placeholder");

  Algorithm* tfp      = factory.create("TensorflowPredict",
                                       "graphFilename", modelName,
                                       "inputs", inputs,
                                       "outputs", outputs,
                                       "isTraining", false,
                                       "isTrainingName", "model/Placeholder_1");
                                   
  Algorithm* ptt      = factory.create("PoolToTensor",
                                       "namespace", "model/Softmax");
                                   
  Algorithm* ttv      = factory.create("TensorToVectorReal");
                                   


  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  audio->output("audio")    >>  fc->input("signal");
  fc->output("frame")       >>  w->input("frame");
  w->output("frame")        >>  spec->input("frame");
  spec->output("spectrum")  >>  mel->input("spectrum");
  mel->output("bands")      >>  logNorm->input("array");
  logNorm->output("array")  >>  vtt->input("frame");
  vtt->output("tensor")     >>  ttp->input("tensor");
  ttp->output("pool")       >>  tfp->input("poolIn");
  tfp->output("poolOut")    >>  ptt->input("pool");
  ptt->output("tensor")     >>  ttv->input("tensor");
  ttv->output("frame")      >>  PC(pool, "vgg_probs"); // store only the mfcc coeffs


  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  // create a network with our algorithms...
  Network n(audio);
  // ...and run it, easy as that!
  n.run();

  // aggregate the results
  Pool aggrPool; // the pool with the aggregated MFCC values
  const char* stats[] = {"mean"};

  standard::Algorithm* aggr = standard::AlgorithmFactory::create("PoolAggregator",
                                                                 "defaultStats", arrayToVector<string>(stats));

  aggr->input("input").set(pool);
  aggr->output("output").set(aggrPool);
  aggr->compute();

  // write results to file
  cout << "-------- writing results to standard output " << outputFilename << " --------" << endl;

  cout << aggrPool.value<vector<Real> >("vgg_probs.mean");

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "format", "json",
                                                                   "filename", outputFilename);
  output->input("pool").set(aggrPool);
  output->compute();

  // NB: we could just wait for the network to go out of scope, but then this would happen
  //     after the call to essentia::shutdown() where the FFTW structs would already have
  //     been freed, so let's just delete everything explicitly now
  n.clear();

  delete aggr;
  delete output;
  essentia::shutdown();

  return 0;
}
