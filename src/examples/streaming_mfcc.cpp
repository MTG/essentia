/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

int main(int argc, char* argv[]) {

  if (argc != 3) {
    cout << "ERROR: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input yaml_output" << endl;
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
                                    "silentFrames", "noise");

  Algorithm* w     = factory.create("Windowing",
                                    "type", "blackmanharris62");

  Algorithm* spec  = factory.create("Spectrum");
  Algorithm* mfcc  = factory.create("MFCC");


  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  // Audio -> FrameCutter
  audio->output("audio")    >>  fc->input("signal");

  // FrameCutter -> Windowing -> Spectrum
  fc->output("frame")       >>  w->input("frame");
  w->output("frame")        >>  spec->input("frame");

  // Spectrum -> MFCC -> Pool
  spec->output("spectrum")  >>  mfcc->input("spectrum");

  mfcc->output("bands")     >>  NOWHERE;                          // we don't want the mel bands
  mfcc->output("mfcc")      >>  PC(pool, "lowlevel.mfcc"); // store only the mfcc coeffs

  // Note: PC is a #define for PoolConnector


  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  // create a network with our algorithms...
  Network n(audio);
  // ...and run it, easy as that!
  n.run();

  // aggregate the results
  Pool aggrPool; // the pool with the aggregated MFCC values
  const char* stats[] = { "mean", "var", "min", "max", "cov", "icov" };

  standard::Algorithm* aggr = standard::AlgorithmFactory::create("PoolAggregator",
                                                                 "defaultStats", arrayToVector<string>(stats));

  aggr->input("input").set(pool);
  aggr->output("output").set(aggrPool);
  aggr->compute();

  aggrPool.merge("lowlevel.mfcc.frames", pool.value<vector<vector<Real> > >("lowlevel.mfcc"));

  // write results to file
  cout << "-------- writing results to file " << outputFilename << " --------" << endl;

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
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
