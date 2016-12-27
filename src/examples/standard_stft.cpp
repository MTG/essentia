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
using namespace std;
using namespace essentia;
using namespace standard;


void scaleAudioVector(vector<Real> &buffer, const Real scale)
{
for (int i=0; i < int(buffer.size()); ++i){
    buffer[i] = scale * buffer[i];
}
}

int main(int argc, char* argv[]) {

  if (argc != 3) {
    cout << "Error: incorrect number of arguments." << endl;
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
  int framesize = 1024;
  int hopsize = 256;
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

  Algorithm* window       = factory.create("Windowing", "type", "hann");

  Algorithm* fft     = factory.create("FFT",
                            "size", framesize);

  Algorithm* ifft     = factory.create("IFFT",
                                "size", framesize);

  Algorithm* overlapAdd = factory.create("OverlapAdd",
                                            "frameSize", framesize,
                                           "hopSize", hopsize);


  Algorithm* audioWriter = factory.create("MonoWriter",
                                     "filename", outputFilename);



  vector<Real> audio;
  vector<Real> frame;
  vector<Real> wframe;
  vector<complex<Real> > fftframe;
  vector<Real> ifftframe;
  vector<Real> alladuio; // concatenated audio file output
 // Real confidence;

  // analysis
  audioLoader->output("audio").set(audio);



  frameCutter->input("signal").set(audio);
  frameCutter->output("frame").set(frame);

  window->input("frame").set(frame);
  window->output("frame").set(wframe);

  fft->input("frame").set(wframe);
  fft->output("fft").set(fftframe);


  // Synthesis
  ifft->input("fft").set(fftframe);
  ifft->output("frame").set(ifftframe);


  vector<Real> audioOutput;

  overlapAdd->input("signal").set(ifftframe); // or frame ?
  overlapAdd->output("signal").set(audioOutput);




////////
/////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  audioLoader->compute();
int counter = 0;
  while (true) {

    // compute a frame
    frameCutter->compute();

    // if it was the last one (ie: it was empty), then we're done.
    if (!frame.size()) {
      break;
    }

    window->compute();
    fft->compute();
    ifft->compute();
    overlapAdd->compute();



    // skip first half window
    if (counter >= floor(framesize / (hopsize * 2.f))){
        alladuio.insert(alladuio.end(), audioOutput.begin(), audioOutput.end());
    }


    counter++;
  }



  // write results to file
  cout << "-------- writing results to file " << outputFilename << " ---------" << endl;

    // write to output file
    audioWriter->input("audio").set(alladuio);
    audioWriter->compute();


  delete audioLoader;
  delete frameCutter;
  delete fft;
  delete ifft;
  delete overlapAdd;
  delete audioWriter;

  essentia::shutdown();

  return 0;
}


