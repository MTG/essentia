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
  Real sampleRate = 16000.0;
  int frameSize = 512;
  int hopSize = 256;
  int patchSize = 187;

  // mel bands parameters
  int numberBands=96;
  string weighting = "linear";
  string warpingFormula = "slaneyMel";
  string normalize = "unit_tri";

  vector<int> inputShape({-1, 1, patchSize, numberBands});
  vector<string> inputs({"model/Placeholder"});
  vector<string> outputs({"model/Sigmoid"});

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
                                    "hopSize", hopSize);

  Algorithm* w     = factory.create("Windowing",
                                    "normalized", false);

  Algorithm* spec  = factory.create("Spectrum",
                                    "size", frameSize);

  Algorithm* mel  = factory.create("MelBands",
                                   "numberBands", numberBands,
                                   "sampleRate", sampleRate,
                                   "highFrequencyBound", sampleRate / 2,
                                   "inputSize", frameSize / 2 + 1,
                                   "weighting", weighting,
                                   "normalize", normalize,
                                   "warpingFormula", warpingFormula);

  Algorithm* shift  = factory.create("UnaryOperator",
                                      "shift", 1,
                                      "scale", 1000);

  Algorithm* comp  = factory.create("UnaryOperator",
                                       "type", "log10");

  Algorithm* vtt      = factory.create("VectorRealToTensor",
                                       "shape", inputShape,
                                       "lastPatchMode", "repeat",
                                       "patchHopSize", patchSize / 2);

  Algorithm* ttp      = factory.create("TensorToPool",
                                       "namespace", "model/Placeholder");

  Algorithm* tfp      = factory.create("TensorflowPredict",
                                       "graphFilename", modelName,
                                       "inputs", inputs,
                                       "outputs", outputs,
                                       "isTraining", false,
                                       "isTrainingName", "model/Placeholder_1");
                                   
  Algorithm* ptt      = factory.create("PoolToTensor",
                                       "namespace", "model/Sigmoid");
                                   
  Algorithm* ttv      = factory.create("TensorToVectorReal");
                                   


  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  audio->output("audio")    >>  fc->input("signal");
  fc->output("frame")       >>  w->input("frame");
  w->output("frame")        >>  spec->input("frame");
  spec->output("spectrum")  >>  mel->input("spectrum");
  mel->output("bands")      >>  shift->input("array");
  shift->output("array")    >>  comp->input("array");
  comp->output("array")     >>  vtt->input("frame");
  vtt->output("tensor")     >>  ttp->input("tensor");
  ttp->output("pool")       >>  tfp->input("poolIn");
  tfp->output("poolOut")    >>  ptt->input("pool");
  ptt->output("tensor")     >>  ttv->input("tensor");
  ttv->output("frame")      >>  PC(pool, "probs"); // store only the mfcc coeffs


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

  cout << aggrPool.value<vector<Real> >("probs.mean");

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
