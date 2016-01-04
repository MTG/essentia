/*
 * Copyright (C) 2006-2015  Music Technology Group - Universitat Pompeu Fabra
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
    cout << "Usage: " << argv[0] << " input_audiofile output_audiofile [1/0 print to stdout]" << endl;
    exit(1);
  }

  // register the algorithms in the factory(ies)
  essentia::init();

  /////// PARAMS //////////////
  int framesize = 1024;
  int hopsize = 256;
  int sr = 44100;

  string audioFilename = argv[1];
  string outputFilename = argv[2];

  // instanciate facgory and create algorithms:
  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audioload    = factory.create("MonoLoader",
                                           "filename", audioFilename,
                                           "sampleRate", sr,
                                           "downmix", "mix");

  Algorithm* frameCutter  = factory.create("FrameCutter",
                                           "frameSize", framesize,
                                           "hopSize", hopsize,
                                           "silentFrames", "noise",
                                           "startFromZero", false );

  Algorithm* window       = factory.create("Windowing", "type", "hann");

  Algorithm* fft     = factory.create("FFT",
                            "size", framesize);

  Algorithm* ifft     = factory.create("IFFT",
                                "size", framesize);

  Algorithm* overlapAdd = factory.create("OverlapAdd",
                                            "frameSize", framesize,
                                           "hopSize", hopsize);

  Algorithm* writer = factory.create("MonoWriter",
                                     "filename", outputFilename);



  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  // audio -> framecutter
  audioload->output("audio")              >>  frameCutter->input("signal");

  // framecutter -> windowing -> fft
  frameCutter->output("frame")            >>  window->input("frame");
  window->output("frame")                 >>  fft->input("frame");

  // fft -> ifft (here possible spectral transformation can be used)

  fft->output("fft")                 >>  ifft->input("fft");
  ifft->output("frame")                 >> overlapAdd->input("frame");
  overlapAdd->output("signal")  >>writer->input("audio");


  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << argv[1]<< " --------" << endl;

  Network network(audioload);
  network.run();

  // write results to yamlfile
  cout << "-------- writing results to wav file " << argv[2] << " --------" << endl;

essentia::shutdown();

  return 0;
}
