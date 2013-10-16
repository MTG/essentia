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
#include <essentia/scheduler/network.h>
#include <essentia/streaming/algorithms/poolstorage.h>

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;


int main(int argc, char* argv[]) {

  if (argc < 3 ) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile output_yamlfile [1/0 print to stdout]" << endl;
    exit(1);
  }

  // register the algorithms in the factory(ies)
  essentia::init();

  /////// PARAMS //////////////
  int framesize = 1024;
  int hopsize = 256;
  int sr = 44100;
  Real tol = 0.1;


  // instanciate facgory and create algorithms:
  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audioload    = factory.create("MonoLoader",
                                           "filename", argv[1],
                                           "sampleRate", sr,
                                           "downmix", "mix");

  Algorithm* frameCutter  = factory.create("FrameCutter",
                                           "frameSize", framesize,
                                           "hopSize", hopsize,
                                           "silentFrames", "noise",
                                           "startFromZero", false );

  Algorithm* window       = factory.create("Windowing", "type", "hann");

  Algorithm* spectrum     = factory.create("Spectrum");

  Algorithm* pitchDetect  = factory.create("PitchYinFFT",
                                           "frameSize", framesize,
                                           "sampleRate", sr);
  // data storage
  Pool pool;

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  // audio -> framecutter
  audioload->output("audio")              >>  frameCutter->input("signal");

  // framecutter -> windowing -> spectrum
  frameCutter->output("frame")            >>  window->input("frame");
  window->output("frame")                 >>  spectrum->input("frame");

  // Spectrum -> pitch detection  -> Pool
  spectrum->output("spectrum")            >>  pitchDetect->input("spectrum");
  pitchDetect->output("pitch")            >>  PC(pool, "tonal.pitch");
  pitchDetect->output("pitchConfidence")  >>  PC(pool, "tonal.pitch confidence");

  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << argv[1]<< " --------" << endl;

  Network(audioload).run();

  // write results to yamlfile
  cout << "-------- writing results to file " << argv[2] << " --------" << endl;

  standard::Algorithm* aggregator = standard::AlgorithmFactory::create("PoolAggregator");

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", argv[2]);

  Pool poolStats;

  aggregator->input("input").set(pool);
  aggregator->output("output").set(poolStats);
  output->input("pool").set(poolStats);

  aggregator->compute();
  output->compute();

  if (argc == 4 && atoi(argv[3])) {
    // printing to stdout:
    const vector<Real>& pitches = pool.value<vector<Real> >("tonal.pitch");
    cout << "number of frames: " << pitches.size() << endl;
    cout << pitches << endl;
  }

  // clean up:
  //deleteNetwork(audioload);
  delete output;
  delete aggregator;
  essentia::shutdown();

  return 0;
}
