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
#include <essentia/pool.h>

#include <essentia/utils/synth_utils.h>

using namespace std;
using namespace essentia;
using namespace standard;


int main(int argc, char* argv[]) {

  if (argc != 3) {
    cout << "Standard_Sinesubtraction Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input output_file" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  /////// PARAMS //////////////

  /////// PARAMS //////////////
  int framesize = 2048;
  int hopsize = 128; //128;
  Real sr = 44100;
  



  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* audioLoader    = factory.create("MonoLoader",
                                           "filename", audioFilename,
                                           "sampleRate", sr,
                                           "downmix", "mix");

  Algorithm* frameCutter  = factory.create("FrameCutter",
                                           "frameSize", framesize,
                                           "hopSize", hopsize,
                                         //  "silentFrames", "noise",
                                           "startFromZero", false );


  // parameters used in the SMS Python implementation

  Algorithm* sinemodelanal     = factory.create("SineModelAnal",
                            "sampleRate", sr,
                            "maxnSines", 100,
                            "freqDevOffset", 10,
                            "freqDevSlope", 0.001
                            );
                            

  
  int subtrFFTSize = std::min(framesize/4, 4*hopsize);  // make sure the FFT size 
  Algorithm* sinesubtraction = factory.create("SineSubtraction",
                              "sampleRate", sr,
                              "fftSize", subtrFFTSize,
                              "hopSize", hopsize
                              );

  Algorithm* audioWriter = factory.create("MonoWriter",
                                     "filename", outputFilename);


  vector<Real> audio;
  vector<Real> frame;

  vector<Real> magnitudes;
  vector<Real> frequencies;
  vector<Real> phases;


  vector<Real> allaudio; // concatenated audio file output


  // analysis
  audioLoader->output("audio").set(audio);

  frameCutter->input("signal").set(audio);
  frameCutter->output("frame").set(frame);

  // Sine model analysis
  sinemodelanal->input("frame").set(frame); // inputs a frame
  sinemodelanal->output("magnitudes").set(magnitudes);
  sinemodelanal->output("frequencies").set(frequencies);
  sinemodelanal->output("phases").set(phases);
  

  vector<Real> audioOutput;

// this needs to take into account overlap-add issues, introducing delay
 sinesubtraction->input("frame").set(frame); // size is iput _fftSize
 sinesubtraction->input("magnitudes").set(magnitudes);
 sinesubtraction->input("frequencies").set(frequencies);
 sinesubtraction->input("phases").set(phases);
 sinesubtraction->output("frame").set(audioOutput); // Nsyn size

////////
/////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  audioLoader->compute();

//-----------------------------------------------
// analysis loop
  cout << "-------- analyzing to sine model parameters" " ---------" << endl;
  int counter = 0;

  while (true) {

    // compute a frame
    frameCutter->compute();

    // if it was the last one (ie: it was empty), then we're done.
    if (!frame.size()) {
      break;
    }

    // Sine model analysis
    sinemodelanal->compute();

    sinesubtraction->compute();

     // skip first half window
    if (counter >= floor(framesize / (hopsize * 2.f))){
        allaudio.insert(allaudio.end(), audioOutput.begin(), audioOutput.end());
    }

    counter++;
  }


  // write results to file
  cout << "-------- writing results to file " << outputFilename << " ---------" << endl;
  cout << "-------- "  << counter<< " frames (hopsize: " << hopsize << ") ---------"<< endl;

    // write to output file
    audioWriter->input("audio").set(allaudio);
    audioWriter->compute();



  delete audioLoader;
  delete frameCutter;
  delete sinemodelanal;
  delete sinesubtraction;
  delete audioWriter;

  essentia::shutdown();

  return 0;
}



