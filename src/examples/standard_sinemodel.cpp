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
    cout << "Standard_SineModel Error: incorrect number of arguments." << endl;
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
  int hopsize = 512;
  Real sr = 44100;
  Real minSineDur = 0.02;

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
  Algorithm* window       = factory.create("Windowing", "type", "hamming");

  Algorithm* fft     = factory.create("FFT",
                            "size", framesize);

  // parameters used in the SMS Python implementation
  Algorithm* sinemodelanal     = factory.create("SineModelAnal",
                            "sampleRate", sr,
                            "maxnSines", 100,
                            "freqDevOffset", 10,
                            "freqDevSlope", 0.001
                            );

  Algorithm* sinemodelsynth     = factory.create("SineModelSynth",
                            "sampleRate", sr, "fftSize", framesize, "hopSize", hopsize);


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

  vector<Real> magnitudes;
  vector<Real> frequencies;
  vector<Real> phases;

  vector<complex<Real> >  sfftframe; // sine model FFT frame
  vector<Real> ifftframe;
  vector<Real> alladuio; // concatenated audio file output
 

  vector< vector<Real> > frequenciesAllFrames;
  vector< vector<Real> > magnitudesAllFrames;
  vector< vector<Real> > phasesAllFrames;

  // analysis
  audioLoader->output("audio").set(audio);

  frameCutter->input("signal").set(audio);
  frameCutter->output("frame").set(frame);

  window->input("frame").set(frame);
  window->output("frame").set(wframe);

  fft->input("frame").set(wframe);
  fft->output("fft").set(fftframe);

  // Sine model analysis
  sinemodelanal->input("fft").set(fftframe);
  sinemodelanal->output("magnitudes").set(magnitudes);
  sinemodelanal->output("frequencies").set(frequencies);
  sinemodelanal->output("phases").set(phases);


  sinemodelsynth->input("magnitudes").set(magnitudes);
  sinemodelsynth->input("frequencies").set(frequencies);
  sinemodelsynth->input("phases").set(phases);
  sinemodelsynth->output("fft").set(sfftframe);

  // Synthesis
  ifft->input("fft").set(sfftframe);
  ifft->output("frame").set(ifftframe);

  vector<Real> audioOutput;

  overlapAdd->input("signal").set(ifftframe);
  overlapAdd->output("signal").set(audioOutput);


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

    window->compute();
    fft->compute();

    // Sine model analysis
    sinemodelanal->compute();

    // append frequencies of the curent frame for later cleaningTracks
    frequenciesAllFrames.push_back(frequencies);
    magnitudesAllFrames.push_back(magnitudes);
    phasesAllFrames.push_back(phases);


    counter++;
  }


  // clean sine tracks
  int minFrames = int( minSineDur * sr / Real(hopsize));
  cleaningSineTracks(frequenciesAllFrames, minFrames);


//-----------------------------------------------
// synthesis loop
  cout << "-------- synthesizing from sine model parameters" " ---------" << endl;
  int nFrames = counter;
  counter = 0;

  while (true) {

    // all frames processed
    if (counter >= nFrames) {
      break;
    }
    // get sine tracks values for the the curent frame, and remove from list
    if (frequenciesAllFrames.size() > 0)
    {
      frequencies = frequenciesAllFrames[0];
      magnitudes = magnitudesAllFrames[0];
      phases = phasesAllFrames[0];
      frequenciesAllFrames.erase (frequenciesAllFrames.begin());
      magnitudesAllFrames.erase (magnitudesAllFrames.begin());
      phasesAllFrames.erase (phasesAllFrames.begin());
    }

    // Sine model synthesis
    sinemodelsynth->compute();

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
  delete sinemodelanal;
  delete sinemodelsynth;
  delete ifft;
  delete overlapAdd;
  delete audioWriter;

  essentia::shutdown();

  return 0;
}


