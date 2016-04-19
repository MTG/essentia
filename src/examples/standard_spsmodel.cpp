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
    cout << "Standard_SPSModel Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input output_file" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];
  string outputSineFilename = outputFilename;
  string outputStocFilename = outputFilename;
  outputSineFilename.replace(outputSineFilename.end()-4,outputSineFilename.end(), "_sine.wav");
  outputStocFilename.replace(outputStocFilename.end()-4,outputStocFilename.end(), "_stoc.wav");

cout << outputSineFilename<< "" << outputStocFilename << endl;
  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  /////// PARAMS //////////////

  /////// PARAMS //////////////
  int framesize = 2048;
  int hopsize = 128; //128;
  Real sr = 44100;
  Real minSineDur = 0.02;
  Real stocf =  0.2; // stochastic envelope factor. Default 0.2


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
  Algorithm* spsmodelanal   = factory.create("SpsModelAnal",
                            "sampleRate", sr,
                            "hopSize", hopsize,
                            "fftSize", framesize,
                            "maxnSines", 100, // 100
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

  vector<complex<Real> >  sfftframe; // sine model FFT frame
  vector<Real> ifftframe;

  vector<Real> allaudio; // concatenated audio file output
  vector<Real> allsineaudio; // concatenated audio file output
  vector<Real> allstocaudio; // concatenated audio file output


  vector< vector<Real> > frequenciesAllFrames;
  vector< vector<Real> > magnitudesAllFrames;
  vector< vector<Real> > phasesAllFrames;
  vector< vector<Real> > stocEnvAllFrames;

  // analysis
  audioLoader->output("audio").set(audio);

  frameCutter->input("signal").set(audio);
  frameCutter->output("frame").set(frame);

  // Sine model analysis
  spsmodelanal->input("frame").set(frame); // inputs a frame
  spsmodelanal->output("magnitudes").set(magnitudes);
  spsmodelanal->output("frequencies").set(frequencies);
  spsmodelanal->output("phases").set(phases);
  spsmodelanal->output("stocenv").set(stocenv);

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
    spsmodelanal->compute();

    // append frequencies of the curent frame for later cleaningTracks
    frequenciesAllFrames.push_back(frequencies);
    magnitudesAllFrames.push_back(magnitudes);
    phasesAllFrames.push_back(phases);
    stocEnvAllFrames.push_back(stocenv);

    counter++;
  }

//-----------------------------------------------
// clean sine tracks
  int minFrames = int( minSineDur * sr / Real(hopsize));
  cleaningSineTracks(frequenciesAllFrames, minFrames);


//-----------------------------------------------
// synthesis loop
  cout << "-------- synthesizing from stochastic model parameters" "----"  << counter<< " frames (hopsize: " << hopsize << ") ---------"<< endl;
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
  delete spsmodelanal;
  delete spsmodelsynth;
  delete audioWriter;
  delete audioWriterSine;
  delete audioWriterStoc;

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


