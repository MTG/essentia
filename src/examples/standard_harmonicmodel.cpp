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

  if (argc < 3) {
    cout << "Standard_HarmonicModel Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input output_file [predominant]. " << endl;
    cout << "\t [predominant]: Optional argument. It is a flag value that can be 0 or 1 .  Set to 1  if PredominantPitchMelodia extraction is used  Default uses Pitch-YinFFT. " << endl;


    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];

   bool usePredominant = false;  
  if   (argc == 4) {
     string usePredominantStr = argv[3];
    if (usePredominantStr == "1")
    {
      usePredominant = true;  
      cout << "Using PredominantPitchMelodia instead of  the default PitchYinFFT."  << endl;
    }
  
  }
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
                  
 Algorithm* equalLoudness = factory.create("EqualLoudness");
                                       
  Algorithm* predominantMelody = factory.create("PredominantPitchMelodia", 
                                                "frameSize", framesize,
                                                "hopSize", hopsize,
                                                "sampleRate", sr);


  // parameters used in the SMS Python implementation
  Algorithm* harmonicmodelanal   = factory.create("HarmonicModelAnal",
                            "sampleRate", sr,
                            "hopSize", hopsize,
                            "nHarmonics", 100,                           
                            "harmDevSlope", 0.01,
                            "maxFrequency", maxF0,
                            "minFrequency", minF0
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
  vector<Real> eqaudio;
  vector<Real> wframe;
  vector<complex<Real> > fftframe;

  vector<Real> predPitch;
  vector<Real> predConf;

  vector<Real> magnitudes;
  vector<Real> frequencies;
  vector<Real> phases;

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

  window->input("frame").set(frame);
  window->output("frame").set(wframe);

  fft->input("frame").set(wframe);
  fft->output("fft").set(fftframe);
  
  // configure spectrum:
  vector<Real> spec;
  spectrum->input("frame").set(wframe);
  spectrum->output("spectrum").set(spec);
  
  equalLoudness->input("signal").set(audio);
  equalLoudness->output("signal").set(eqaudio);


// PREDOMINANT  pitch analysis
  predominantMelody->input("signal").set(eqaudio);
  predominantMelody->output("pitch").set(predPitch);
  predominantMelody->output("pitchConfidence").set(predConf);

  Real thisPitch = 0. ,thisConf = 0;

  pitchDetect->input("spectrum").set(spec);
  pitchDetect->output("pitch").set(thisPitch);
  pitchDetect->output("pitchConfidence").set(thisConf);
   
  // Harmonic model analysis
  harmonicmodelanal->input("fft").set(fftframe); // input fft
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
      fft->compute();
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
  delete   window;
  delete   fft;
  delete  spectrum;
  delete equalLoudness; 
  delete  pitchDetect; 
   delete harmonicmodelanal;
	delete predominantMelody;
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


