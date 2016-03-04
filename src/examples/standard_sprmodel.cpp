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
    cout << "Standard_SprModel ERROR: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input output_file" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];
  string outputSineFilename = outputFilename;
  string outputResFilename = outputFilename;
  outputSineFilename.replace(outputSineFilename.end()-4,outputSineFilename.end(), "_sine.wav");
  outputResFilename.replace(outputResFilename.end()-4,outputResFilename.end(), "_res.wav");

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;




  /////// PARAMS //////////////
  int framesize = 2048;
  int hopsize = 128; 
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
  Algorithm* sprmodelanal   = factory.create("SprModelAnal",
                            "sampleRate", sr,
                            "hopSize", hopsize,
                            "fftSize", framesize,
                            "freqDevOffset", 10,
                            "freqDevSlope", 0.001       
                            );

  
   Algorithm* sprmodelsynth     = factory.create("SprModelSynth",
                            "sampleRate", sr, "fftSize", framesize, "hopSize", hopsize);
   
  Algorithm* audioWriter = factory.create("MonoWriter",
                                     "filename", outputFilename);
  Algorithm* audioWriterSine = factory.create("MonoWriter",
                                     "filename", outputSineFilename);
  Algorithm* audioWriterRes = factory.create("MonoWriter",
                                     "filename", outputResFilename);
                                     
 
  vector<Real> audio;
  vector<Real> frame;

  vector<Real> magnitudes;
  vector<Real> frequencies;
  vector<Real> phases;
  vector<Real> res;


  vector<Real> allaudio; // concatenated audio file output
  vector<Real> allsineaudio; // concatenated audio file output
  vector<Real> allresaudio; // concatenated audio file output
  
  
  // accumulate estimated values   for all frames for cleaning tracks before synthesis
  vector< vector<Real> > frequenciesAllFrames;
  vector< vector<Real> > magnitudesAllFrames;
  vector< vector<Real> > phasesAllFrames;
  vector< vector<Real> > resAllFrames;

  vector<complex<Real> >  sfftframe; // sine model FFT frame
  vector<Real> ifftframe; //  sine model IFFT frame


  // analysis
  audioLoader->output("audio").set(audio);

  frameCutter->input("signal").set(audio);
  frameCutter->output("frame").set(frame);

   
  // Harmonic model analysis
  sprmodelanal->input("frame").set(frame); // inputs a frame
  sprmodelanal->output("magnitudes").set(magnitudes);
  sprmodelanal->output("frequencies").set(frequencies);
  sprmodelanal->output("phases").set(phases);
  sprmodelanal->output("res").set(res);
  

 vector<Real> audioOutput;
  vector<Real> audioSineOutput;
  vector<Real> audioResOutput;

// Sinusoidal Model Synthesis (only harmonics)
  sprmodelsynth->input("magnitudes").set(magnitudes);
  sprmodelsynth->input("frequencies").set(frequencies);
  sprmodelsynth->input("phases").set(phases);
  sprmodelsynth->input("res").set(res);
  sprmodelsynth->output("frame").set(audioOutput);
  sprmodelsynth->output("sineframe").set(audioSineOutput);
  sprmodelsynth->output("resframe").set(audioResOutput);



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


    // HpS model analysis
    sprmodelanal->compute();
     
    
    // append frequencies of the curent frame for later cleaningTracks
    frequenciesAllFrames.push_back(frequencies);
    magnitudesAllFrames.push_back(magnitudes);
    phasesAllFrames.push_back(phases);
    resAllFrames.push_back(res);
    
    counter++;
  }
  

  // clean sine tracks
  int minFrames = int( minSineDur * sr / Real(hopsize));
  cleaningSineTracks(frequenciesAllFrames, minFrames);


//-----------------------------------------------
// synthesis loop
  cout << "-------- synthesizing from sinusoidal plus residual model parameters" " ---------" << endl;
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
      res =resAllFrames[0];
      frequenciesAllFrames.erase (frequenciesAllFrames.begin());
      magnitudesAllFrames.erase (magnitudesAllFrames.begin());
      phasesAllFrames.erase (phasesAllFrames.begin());
      resAllFrames.erase (resAllFrames.begin());
    }


    // SPR model synthesis
    sprmodelsynth->compute();


    
    // skip first half window
    if (counter >= floor(framesize / (hopsize * 2.f))){
       allaudio.insert(allaudio.end(), audioOutput.begin(), audioOutput.end());
       allsineaudio.insert(allsineaudio.end(), audioSineOutput.begin(), audioSineOutput.end());
       allresaudio.insert(allresaudio.end(), audioResOutput.begin(), audioResOutput.end());
    }

    counter++;
  }



  // write results to file
  cout << "-------- writing results to file " << outputFilename << " ---------" << endl;
  cout << "-------- "  << counter<< " frames (hopsize: " << hopsize << ") ---------"<< endl;

  // write to output file
  audioWriter->input("audio").set(allaudio);  
  audioWriter->compute();

  // write sinusoidal and residual components
  audioWriterSine->input("audio").set(allsineaudio);
  audioWriterSine->compute();

  audioWriterRes->input("audio").set(allresaudio);
  audioWriterRes->compute();

  delete audioLoader;
  delete frameCutter;
  delete sprmodelanal;
  delete sprmodelsynth;
  delete audioWriter;
  delete audioWriterSine;
  delete audioWriterRes;
  
  essentia::shutdown();

  return 0;
}

