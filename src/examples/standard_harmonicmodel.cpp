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

std::vector< std::vector<Real> > readIn2dData(const char* filename);

int main(int argc, char* argv[]) {

  if (argc != 3) {
    cout << "Standard_HarmonicModel ERROR: incorrect number of arguments." << endl;
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
  
   bool usePredominant = true; // set to true if PredmonantMelody extraction is used. Set to false if monhonic Pitch-YinFFT is used,


 Real minSineDur = 0.02;
 //Real stocf = 0.2; // 0.2; //1.; // stochastic envelope factor. Default 0.2


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


 Algorithm* equalLoudness = factory.create("EqualLoudness");


  Algorithm* window      = factory.create("Windowing", "type", "hann");

  Algorithm* spectrum = factory.create("Spectrum",
                                       "size", framesize);

  Algorithm* pitchDetect = factory.create("PitchYinFFT",
                                          "frameSize", framesize,
                                          "sampleRate", sr);
                                       
  Algorithm* predominantMelody = factory.create("PredominantPitchMelodia", //PredominantMelody",
                                                "frameSize", framesize,
                                                "hopSize", hopsize,
                                                "sampleRate", sr);

    
  // parameters used in the SMS Python implementation
  Algorithm* harmonicmodelanal   = factory.create("HarmonicModelAnal",
                            "sampleRate", sr,
                            "hopSize", hopsize,
                            "fftSize", framesize,
                            "nHarmonics", 100,                           
                            "harmDevSlope", 0.01
                            );


//  printf("TODO: replace this workflow (harmonicdemodelanal and sinesubtraction)     by a call to    HprModelAnal algorithm\n");

/*  int subtrFFTSize = std::min(512, 4*hopsize);  // make sure the FFT size is at least twice the hopsize
  Algorithm* sinesubtraction = factory.create("SineSubtraction",
                              "sampleRate", sr,
                              "fftSize", subtrFFTSize,
                              "hopSize", hopsize
                              );*/


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
  vector<Real> eqaudio;
  vector<Real> wframe;

  vector<Real> predPitch;
  vector<Real> predConf;

  vector<Real> magnitudes;
  vector<Real> frequencies;
  vector<Real> phases;
  vector<Real> stocenv;

  // accumulate estimated values   for all frames for cleaning tracks before synthesis
  vector< vector<Real> > frequenciesAllFrames;
  vector< vector<Real> > magnitudesAllFrames;
  vector< vector<Real> > phasesAllFrames;

  vector<complex<Real> >  sfftframe; // sine model FFT frame
  vector<Real> ifftframe; //  sine model IFFT frame
  vector<Real> allaudio; // concatenated audio file output


  // analysis
  audioLoader->output("audio").set(audio);

  frameCutter->input("signal").set(audio);
  frameCutter->output("frame").set(frame);

  equalLoudness->input("signal").set(audio);
  equalLoudness->output("signal").set(eqaudio);


// PREDOMINANT  pitch analysis
  predominantMelody->input("signal").set(eqaudio);
  predominantMelody->output("pitch").set(predPitch);
  predominantMelody->output("pitchConfidence").set(predConf);

// YIN pitch analysis
 window->input("frame").set(frame);
  window->output("frame").set(wframe);

  // set spectrum:
  vector<Real> spec;
  spectrum->input("frame").set(wframe);
  spectrum->output("spectrum").set(spec);

  // set Yin pitch extraction:
  Real thisPitch = 0.;
  Real thisConf = 0;
  pitchDetect->input("spectrum").set(spec);
  pitchDetect->output("pitch").set(thisPitch);
  pitchDetect->output("pitchConfidence").set(thisConf);
  
  
  // Harmonic model analysis
  harmonicmodelanal->input("frame").set(frame); // inputs a frame
  harmonicmodelanal->input("pitch").set(thisPitch); // inputs a pitch
  harmonicmodelanal->output("magnitudes").set(magnitudes);
  harmonicmodelanal->output("frequencies").set(frequencies);
  harmonicmodelanal->output("phases").set(phases);
  

// Sinusoidal Model Synthesis (only harmonics)
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




/////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  audioLoader->compute();
  equalLoudness->compute();
  predominantMelody->compute();


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

    // pitch extraction
    spectrum->compute();
    pitchDetect->compute();

    // get predominant pitch
    if (usePredominant){
      thisPitch = predPitch[counter];
    }


    // Harmonic model analysis
    harmonicmodelanal->compute();

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
  delete window;
  delete harmonicmodelanal;
	delete predominantMelody;
  delete pitchDetect;
  delete sinemodelsynth;
  delete audioWriter;

  essentia::shutdown();

  return 0;
}


// support functinos to read data from numpy
std::vector< std::vector<Real> > readIn2dData(const char* filename)
{
    /* Function takes a char* filename argument and returns a
     * 2d dynamic array containing the data
     */

    std::vector< std::vector<Real> > table;
    std::fstream ifs;

    /*  open file  */
    ifs.open(filename);

    while (true)
    {
        std::string line;
        Real buf;
        getline(ifs, line);

        std::stringstream ss(line, std::ios_base::out|std::ios_base::in|std::ios_base::binary);

        if (!ifs)
            // mainly catch EOF
            break;

        if (line[0] == '#' || line.empty())
            // catch empty lines or comment lines
            continue;


        std::vector<Real> row;

        while (ss >> buf)
            row.push_back(buf);

std::cout << "row size from numpy is: " << row.size() << std::endl;
        table.push_back(row);

    }

    ifs.close();

    return table;
}


