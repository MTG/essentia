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
#include <fstream>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/pool.h>
#include "credit_libav.h"

using namespace std;
using namespace essentia;
using namespace standard;

int main(int argc, char* argv[]) {

  // input correct?
  if (argc != 4) {
    cout << "Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input dimensions yaml_output" << endl;
    creditLibAV();
    exit(1);
  }

  string audioFilename = argv[1];
  uint dimensions = atoi(argv[2]);
  string outputFilename = argv[3];

  // register the algorithms in the factory
  essentia::init();


 /////// PARAMS //////////////
  int sampleRate = 44100;
  int frameSize = 2048;
  int hopSize = 1024;

  // we want to compute the MFCC of a file: we need the create the following:
  // audioloader -> framecutter -> windowing -> FFT -> MFCC

  AlgorithmFactory& factory = standard::AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate);

  Algorithm* fc    = factory.create("FrameCutter",
                                    "frameSize", frameSize,
                                    "hopSize", hopSize);

  Algorithm* w     = factory.create("Windowing",
                                    "type", "blackmanharris62");

  Algorithm* spec  = factory.create("Spectrum");
  Algorithm* mfcc  = factory.create("MFCC");

  Algorithm* pca   = factory.create("PCA",
                                    "namespaceIn", "feats",
                                    "namespaceOut", "pca_mfcc",
                                    "dimensions", dimensions);


  // storage
  Pool pool;
  Pool outPool;


  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos ---------" << endl;

  // Audio -> FrameCutter
  vector<Real> audioBuffer;

  audio->output("audio").set(audioBuffer);
  fc->input("signal").set(audioBuffer);

  // FrameCutter -> Windowing -> Spectrum
  vector<Real> frame, windowedFrame;

  fc->output("frame").set(frame);
  w->input("frame").set(frame);

  w->output("frame").set(windowedFrame);
  spec->input("frame").set(windowedFrame);

  // Spectrum -> MFCC
  vector<Real> spectrum, mfccCoeffs, mfccBands;

  spec->output("spectrum").set(spectrum);
  mfcc->input("spectrum").set(spectrum);

  mfcc->output("bands").set(mfccBands);
  mfcc->output("mfcc").set(mfccCoeffs);



  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  audio->compute();

  while (true) {

    // compute a frame
    fc->compute();

    // if it was the last one (ie: it was empty), then we're done.
    if (!frame.size()) {
      break;
    }

    // if the frame is silent, just drop it and go on processing
    if (isSilent(frame)) continue;

    w->compute();
    spec->compute();
    mfcc->compute();

    pool.add("feats", mfccCoeffs);

  }

  pca->input("poolIn").set(pool);
  pca->output("poolOut").set(outPool);

  pca->compute();

  // write results to file
  cout << "-------- writing results to file " << outputFilename << " ---------" << endl;

  Algorithm* output = AlgorithmFactory::create("YamlOutput",
                                               "filename", outputFilename);
  output->input("pool").set(outPool);
  output->compute();

  // clean up
  delete pca;
  delete audio;
  delete fc;
  delete w;
  delete spec;
  delete mfcc;
  delete output;

  essentia::shutdown();

  return 0;
}
