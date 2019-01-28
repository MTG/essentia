/*
 * Copyright (C) 2006-2018  Music Technology Group - Universitat Pompeu Fabra
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
  int hopSize = 256;

  // we want to compute the probabilistic yin pitch track of a file: 
  // we need the create the following:
  // audioloader -> pitchYinProbablistic -> PoolStorage
  // we also need a DevNull which is able to gobble data without doing anything
  // with it (required otherwise a buffer would be filled and blocking)

  AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate);

  Algorithm* yp    = factory.create("PitchYinProbabilistic",
                                    "sampleRate", sampleRate,
                                    "frameSize", frameSize,
                                    "hopSize", hopSize,
                                    "lowAmp", 0.1,
                                    "outputUnvoiced", 2);

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  // Audio -> PitchYinProbabilistic
  audio->output("audio")    >>  yp->input("signal");
  yp->output("pitch")      >>  PC(pool, "tonal.pitch");

  // Note: PC is a #define for PoolConnector

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
