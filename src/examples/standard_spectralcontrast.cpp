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
#include <fstream>
#include <essentia/algorithmfactory.h>
#include <essentia/pool.h>
#include <essentia/essentiamath.h>

using namespace std;
using namespace essentia;
using namespace standard;

int main(int argc, char* argv[]) {

  if (argc != 3) {
    cout << "ERROR: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input yaml_output" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];

  essentia::init();

  /********** SETUP ALGORITHMS **********/

  int frameSize  = 2048;
  int hopSize    = 1024;
  int sampleRate = 44100;

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* audio      = factory.create("EqloudLoader",
                                         "filename", audioFilename,
                                         "sampleRate", sampleRate,
                                         "replayGain", -6.0);

  Algorithm* fcutter    = factory.create("FrameCutter",
                                         "frameSize", frameSize,
                                         "hopSize", hopSize);

  Algorithm* window     = factory.create("Windowing",
                                         "type", "blackmanharris62");

  Algorithm* fft        = factory.create("Spectrum");

  Algorithm* sc         = factory.create("SpectralContrast",
                                         "frameSize", frameSize,
                                         "sampleRate", sampleRate,
                                         "numberBands", 6,
                                         "lowFrequencyBound", 20,
                                         "highFrequencyBound", 11000,
                                         "neighbourRatio", 0.4,
                                         "staticDistribution", 0.15);

  /********** SETUP CONNECTIONS **********/

  vector<Real> audioBuffer;
  audio->output("audio").set(audioBuffer);
  fcutter->input("signal").set(audioBuffer);

  vector<Real> frame, windowedFrame;

  fcutter->output("frame").set(frame);
  window->input("frame").set(frame);

  window->output("frame").set(windowedFrame);
  fft->input("frame").set(windowedFrame);

  vector<Real> spectrum;

  fft->output("spectrum").set(spectrum);
  sc->input("spectrum").set(spectrum);

  vector<Real> sccoeffs;
  vector<Real> scvalleys;

  sc->output("spectralContrast").set(sccoeffs);
  sc->output("spectralValley").set(scvalleys);


  /********** COMPUTATION **********/

  audio->compute();

  Pool poolSc, poolOut;

  /**** frame by frame ****/
  while (true) {

    // get a single frame
    fcutter->compute();

    // if it was the last one (ie: it was empty), then we're done.
    if (!frame.size())
      break;

    // if the frame is silent, just drop it and go on processing
    if (isSilent(frame)) continue;

    // compute
    window->compute();
    fft->compute();
    sc->compute();

    poolSc.add("contrast_coeffs", sccoeffs);
    poolSc.add("contrast_valleys", scvalleys);
  }

  poolOut.add("contrast_coeffs.means", meanFrames(poolSc.value<vector<vector<Real> > >("contrast_coeffs")));
  poolOut.add("contrast_coeffs.vars" , varianceFrames(poolSc.value<vector<vector<Real> > >("contrast_coeffs")));
  poolOut.add("contrast_valleys.means", meanFrames(poolSc.value<vector<vector<Real> > >("contrast_valleys")));
  poolOut.add("contrast_valleys.vars" , varianceFrames(poolSc.value<vector<vector<Real> > >("contrast_valleys")));

  // write yaml file
  Algorithm* output = AlgorithmFactory::create("YamlOutput", "filename", outputFilename);
  output->input("pool").set(poolOut);
  output->compute();

  // clean up
  delete audio;
  delete fcutter;
  delete window;
  delete fft;
  delete sc;
  delete output;

  essentia::shutdown();

  return 0;
}
