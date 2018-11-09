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
#include <essentia/streaming/algorithms/fileoutput.h>
#include <essentia/scheduler/network.h>
#include "credit_libav.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

// This example shows how to use the Tensorflow wrapper of Essentia
// in real time. Here we are running a tensorflow clasiffication model
// build on top of Mel bands. 
int main(int argc, char* argv[]) {

  if (argc != 3) {
    cout << "Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input yaml_output" << endl;
    creditLibAV();
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  /////// PARAMS //////////////
  Real sampleRate = 44100.0;
  int frameSize = 2048;
  int hopSize = 1024;

  AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate);

  Algorithm* fc    = factory.create("FrameCutter",
                                    "frameSize", frameSize,
                                    "hopSize", hopSize,
                                    "silentFrames", "noise");

  Algorithm* w     = factory.create("Windowing",
                                    "type", "blackmanharris62");

  Algorithm* spec  = factory.create("Spectrum");

  Algorithm* mel   = factory.create("MelBands", 
                                    "numberBands", 90);
  
  // VectorRealToTensor creates tensor by accumulating frames
  // and the input shape required by our model is reached. 
  // We can choose the axis to store our features with the 
  // parameter 'timeAxis'
  vector<int> outputShape = {1, 3000, 90};
  Algorithm* vtt   = factory.create("VectorRealToTensor",
                                   "shape", outputShape);

  // The input tensors have to be stores inside an Essentia pool
  // under a namespace that matches the Tensorflow model input 
  // layer name.
  Algorithm* ttp   = factory.create("TensorToPool",
                                   "mode", "overwrite",
                                   "namespace", "bidirectional_1_input_1");

  vector<string> inputs = {"bidirectional_1_input_1"};
  vector<string> outputs = {"output_node0", "time_distributed_1_1/Reshape_1"};

  //  Here is where the Deep Learning magic happens!  
  Algorithm* tfp  = factory.create("TensorflowPredict",
                                   "inputs", inputs,
                                   "outputs", outputs);

  // Here we retrieve the tensors of interest from the output stream of
  // pools.
  Algorithm* ptt   = factory.create("PoolToTensor",
                                    "namespace", "output_node0");

  // In the same manner, the output frame are extracted from the tensors
  // so they can be stored to a file or saved in an storage pool. 
  Algorithm* ttv   = factory.create("TensorToVectorReal",
                                    "shape", outputShape);

  Algorithm* file = new FileOutput<vector<Real> >();
  file->configure("filename", "predictionframes.txt",
                  "mode", "text");

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  audio->output("audio")    >>  fc->input("signal");

  fc->output("frame")       >>  w->input("frame");

  w->output("frame")        >>  spec->input("frame");

  spec->output("spectrum")  >>  mel->input("spectrum");

  mel->output("bands")      >>  vtt->input("frame");

  vtt->output("tensor")     >>  ttp->input("tensor");

  ttp->output("pool")       >>  tfp->input("poolIn");

  tfp->output("poolOut")    >>  ptt->input("pool");

  ptt->output("tensor")     >>  ttv->input("tensor");

  ttv->output("frame")      >>  file->input("data");



  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  // create a network with our algorithms...
  Network n(audio);
  // ...and run it, easy as that!
  n.run();




  // NB: we could just wait for the network to go out of scope, but then this would happen
  //     after the call to essentia::shutdown() where the FFTW structs would already have
  //     been freed, so let's just delete everything explicitly now
  n.clear();

  essentia::shutdown();

  return 0;
}
