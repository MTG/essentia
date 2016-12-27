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
    cout << "Standard_HpSModel Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input output_file" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];
  string outputSineFilename = outputFilename;
  string outputStocFilename = outputFilename;
  outputSineFilename.replace(outputSineFilename.end()-4,outputSineFilename.end(), "_sine.wav");
  outputStocFilename.replace(outputStocFilename.end()-4,outputStocFilename.end(), "_stoc.wav");

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;


  /////// PARAMS //////////////

  /////// PARAMS //////////////
  int framesize = 2048;
  int hopsize = 128; //128;
  Real sr = 44100;
    
  Real minF0 = 65.;
  Real maxF0 = 550.;

 Real minSineDur = 0.02;
 Real stocf = 0.2;  // stochastic envelope factor. Default 0.2
  

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

  Algorithm* window       = factory.create("Windowing", "type", "hamming");

  Algorithm* fft     = factory.create("FFT",
                            "size", framesize);

  Algorithm* spectrum = factory.create("Spectrum",
                                       "size", framesize);

  Algorithm* pitchDetect = factory.create("PitchYinFFT",
                                          "frameSize", framesize,
                                          "sampleRate", sr);

  // parameters used in the SMS Python implementation
  Algorithm* hpsmodelanal   = factory.create("HpsModelAnal",
                            "sampleRate", sr,
                            "hopSize", hopsize,
                            "fftSize", framesize,
                            "maxFrequency", maxF0,
                            "minFrequency", minF0,
                            "nHarmonics", 100,                           
                            "harmDevSlope", 0.01,
                            "freqDevOffset", 10,
                            "freqDevSlope", 0.001,
                            "stocf", stocf
                            );


  Algorithm* spsmodelsynth  = factory.create("SpsModelSynth",
                            "sampleRate", sr, "fftSize", framesize, "hopSize", hopsize, "stocf", stocf);
 
  
  Algorithm* audioWriter = factory.create("MonoWriter",
                                     "filename", outputFilename);
  Algorithm* audioWriterSine = factory.create("MonoWriter",
                                     "filename", outputSineFilename);
  Algorithm* audioWriterStoc = factory.create("MonoWriter",
                                     "filename", outputStocFilename);
                                     
 
  vector<Real> audio;
  vector<Real> frame;
  vector<Real> wframe;
  vector<complex<Real> > fftframe;
  
  vector<Real> magnitudes;
  vector<Real> frequencies;
  vector<Real> phases;
  vector<Real> stocenv;


  vector<Real> allaudio; // concatenated audio file output
  vector<Real> allsineaudio; // concatenated audio file output
  vector<Real> allstocaudio; // concatenated audio file output
  
  
  // accumulate estimated values   for all frames for cleaning tracks before synthesis
  vector< vector<Real> > frequenciesAllFrames;
  vector< vector<Real> > magnitudesAllFrames;
  vector< vector<Real> > phasesAllFrames;
  vector< vector<Real> > stocEnvAllFrames;


  // analysis
  audioLoader->output("audio").set(audio);

  frameCutter->input("signal").set(audio);
  frameCutter->output("frame").set(frame);

   window->input("frame").set(frame);
  window->output("frame").set(wframe);

  fft->input("frame").set(wframe);
  fft->output("fft").set(fftframe);
  
  // configure spectrum:
  vector<Real> spec;
  spectrum->input("frame").set(wframe);
  spectrum->output("spectrum").set(spec);  

  Real thisPitch = 0. ,thisConf = 0;
  pitchDetect->input("spectrum").set(spec);
  pitchDetect->output("pitch").set(thisPitch);
  pitchDetect->output("pitchConfidence").set(thisConf);
   
   
  // Harmonic model analysis
  hpsmodelanal->input("frame").set(frame); // inputs a frame
  hpsmodelanal->input("pitch").set(thisPitch); // inputs a pitch
  hpsmodelanal->output("magnitudes").set(magnitudes);
  hpsmodelanal->output("frequencies").set(frequencies);
  hpsmodelanal->output("phases").set(phases);
  hpsmodelanal->output("stocenv").set(stocenv);
  

 vector<Real> audioOutput;
  vector<Real> audioSineOutput;
  vector<Real> audioStocOutput;

  spsmodelsynth->input("magnitudes").set(magnitudes);
  spsmodelsynth->input("frequencies").set(frequencies);
  spsmodelsynth->input("phases").set(phases);
  spsmodelsynth->input("stocenv").set(stocenv);
  spsmodelsynth->output("frame").set(audioOutput); // outputs a frame
  spsmodelsynth->output("sineframe").set(audioSineOutput); // outputs a frame
  spsmodelsynth->output("stocframe").set(audioStocOutput); // outputs a frame


/////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  audioLoader->compute();

//-----------------------------------------------
// analysis loop
  cout << "-------- analyzing to harmonic model parameters" " ---------" << endl;
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
     spectrum->compute();
     pitchDetect->compute();

    // HpS model analysis
    hpsmodelanal->compute();
     
    
    // append frequencies of the curent frame for later cleaningTracks
    frequenciesAllFrames.push_back(frequencies);
    magnitudesAllFrames.push_back(magnitudes);
    phasesAllFrames.push_back(phases);
    stocEnvAllFrames.push_back(stocenv);
    
    counter++;
  }
  

  // clean sine tracks
  int minFrames = int( minSineDur * sr / Real(hopsize));
  cleaningSineTracks(frequenciesAllFrames, minFrames);


//-----------------------------------------------
// synthesis loop
  cout << "-------- synthesizing from harmonic model parameters" " ---------" << endl;
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
      stocenv = stocEnvAllFrames[0];
      frequenciesAllFrames.erase (frequenciesAllFrames.begin());
      magnitudesAllFrames.erase (magnitudesAllFrames.begin());
      phasesAllFrames.erase (phasesAllFrames.begin());
      stocEnvAllFrames.erase (stocEnvAllFrames.begin());
    }


    // Sine model synthesis
    spsmodelsynth->compute();

    // skip first half window
    if (counter >= floor(framesize / (hopsize * 2.f))){
       allaudio.insert(allaudio.end(), audioOutput.begin(), audioOutput.end());
       allsineaudio.insert(allsineaudio.end(), audioSineOutput.begin(), audioSineOutput.end());
       allstocaudio.insert(allstocaudio.end(), audioStocOutput.begin(), audioStocOutput.end());
    }

    counter++;
  }




  // write results to file
  cout << "-------- writing results to file " << outputFilename << " ---------" << endl;
  cout << "-------- "  << counter<< " frames (hopsize: " << hopsize << ") ---------"<< endl;

  // write to output file
  audioWriter->input("audio").set(allaudio);
  audioWriter->compute();

  // write sinusoidal and stochastic components
  audioWriterSine->input("audio").set(allsineaudio);
  audioWriterSine->compute();


  audioWriterStoc->input("audio").set(allstocaudio);
  audioWriterStoc->compute();

  delete audioLoader;
  delete frameCutter;
  delete   window;
  delete   fft;
  delete  spectrum;  
  delete  pitchDetect; 
  delete hpsmodelanal;
  delete spsmodelsynth;
  delete audioWriter;
  delete audioWriterSine;
  delete audioWriterStoc;
  
  essentia::shutdown();

  return 0;
}

