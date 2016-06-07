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
    cout << "Standard_StochasticModel Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input output_file" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  string outputStocFilename = argv[2];

  cout  << outputStocFilename << endl;
  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  /////// PARAMS //////////////

  /////// PARAMS //////////////

  int hopsize = 256; //128;
  int framesize = 2 * hopsize; // for stochastic analysis/synthesis
  Real sr = 44100;
  Real stocf = 0.1; //0.2; // 0.2; //1.; // stochastic envelope factor. Default 0.2


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
  Algorithm* stocmodelanal   = factory.create("StochasticModelAnal",
                            "sampleRate", sr,
                            "hopSize", hopsize,
                            "fftSize", framesize,
                            "stocf", stocf
                            );

  Algorithm* stocmodelsynth  = factory.create("StochasticModelSynth",
                            "sampleRate", sr,
                            "fftSize", framesize,
                            "hopSize", hopsize,
                            "stocf", stocf);


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

  // analysis
  audioLoader->output("audio").set(audio);

  frameCutter->input("signal").set(audio);
  frameCutter->output("frame").set(frame);

  // stochastic model analysis
  stocmodelanal->input("frame").set(frame); // inputs a frame
  stocmodelanal->output("stocenv").set(stocenv);

  // Synthesis
  vector<Real> audioStocOutput;
  stocmodelsynth->input("stocenv").set(stocenv);
  stocmodelsynth->output("frame").set(audioStocOutput); // outputs a frame


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
    stocmodelanal->compute();
    stocmodelsynth->compute();

    // skip first half window
    if (counter >= floor(framesize / (hopsize * 2.f))){
       allstocaudio.insert(allstocaudio.end(), audioStocOutput.begin(), audioStocOutput.end());
    }

    counter++;
  }


  // write results to file
  cout << "-------- writing results to file " << outputStocFilename << " ---------" << endl;
  cout << "-------- "  << counter<< " frames (hopsize: " << hopsize << ") ---------"<< endl;

  // write to output file
  audioWriterStoc->input("audio").set(allstocaudio);
  audioWriterStoc->compute();

  delete audioLoader;
  delete frameCutter;
  delete stocmodelanal;
  delete stocmodelsynth;


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


