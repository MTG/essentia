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
#include <essentia/algorithmfactory.h>
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/streaming/algorithms/fileoutput.h>
#include <essentia/scheduler/network.h>
#include "credit_libav.h"
#include "music_extractor/extractor_utils.h"


using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;


void usage(char *progname) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << progname << " input_audiofile output_file [profile]" << endl;
    cout << endl << "This extractor computes magnitude spectrogram, mel-band log-energies spectrogram, and mfcc frames for the input audiofile." << endl;
    cout << "The results are stored in either binary files (default) or a yaml/json file. Use the yaml profile file to specify the output:" << endl;
    cout << "- 'outputFormat: binary' for binary output." << endl;
    cout << "- 'outputFormat: json' for json output" << endl;
    cout << "- 'outputFormat: yaml' for yaml output" << endl;

    creditLibAV();

    exit(1);
}


void setDefaultOptions(Pool &options) {
  options.set("sampleRate", 44100.0);
  options.set("outputFormat", "binary");
  options.set("frameSize", 2048);
  options.set("hopSize", 1024);
}


int main(int argc, char* argv[]) {

  string audioFilename, outputFilename, profileFilename;

  switch (argc) {
    case 3:
      audioFilename =  argv[1];
      outputFilename = argv[2];
      break;
    case 4: // profile supplied
      audioFilename =  argv[1];
      outputFilename = argv[2];
      profileFilename = argv[3];
      break;
    default:
      usage(argv[0]);
  }

  essentia::init();

  Pool options;
  Pool pool;
  setDefaultOptions(options);
  setExtractorOptions(profileFilename, options);

  Real sampleRate = options.value<Real>("sampleRate");
  int frameSize = options.value<Real>("frameSize");
  int hopSize = options.value<Real>("hopSize");
  std::string format = options.value<string>("outputFormat");

  AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate);

  Algorithm* fc = factory.create("FrameCutter",
                                 "frameSize", frameSize,
                                 "hopSize", hopSize,
                                 "silentFrames", "noise");

  Algorithm* w = factory.create("Windowing",
                                "type", "blackmanharris62");

  Algorithm* spec  = factory.create("Spectrum");
  Algorithm* mfcc  = factory.create("MFCC", "numberBands", 40,
                                    "numberCoefficients", 20);
  Algorithm* melbands96 = factory.create("MelBands",
                                         "numberBands", 96,
                                         "log", true);

  // Audio -> FrameCutter -> Windowing -> Spectrum -> MelBands & MFCC 
  audio->output("audio") >> fc->input("signal");
  fc->output("frame") >> w->input("frame");
  w->output("frame") >> spec->input("frame");
  spec->output("spectrum") >> mfcc->input("spectrum");
  spec->output("spectrum") >> melbands96->input("spectrum");
  mfcc->output("bands") >> NOWHERE; // we'll only store high-res mel96 bands

  // Prepare binary file outputs or pool/yaml outputs 
  if (format=="binary") {
    Algorithm* fileSpec = new FileOutput<vector<Real> >();
    Algorithm* fileMelBands = new FileOutput<vector<Real> >();
    Algorithm* fileMFCC = new FileOutput<vector<Real> >();

    fileSpec->configure("filename", outputFilename + ".spec", "mode", "binary");
    fileMelBands->configure("filename", outputFilename + ".mel96", "mode", "binary");
    fileMFCC->configure("filename", outputFilename + ".mfcc", "mode", "binary");
 
    melbands96->output("bands") >> fileMelBands->input("data");
    mfcc->output("mfcc") >> fileMFCC->input("data");
    spec->output("spectrum") >> fileSpec->input("data"); 
  }
  else {
    mfcc->output("mfcc") >> PC(pool, "lowlevel.mfcc");
    melbands96->output("bands") >> PC(pool, "lowlevel.melbands96");   
    spec-> output("spectrum") >> PC(pool, "lowlevel.spectrum");
  }

  cout << "Analyzing " << audioFilename << endl;;

  Network n(audio);
  n.run();

  // Write results to file
  if (format!="binary") {
    standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                     "filename", outputFilename,
                                                                     "format", format);
    output->input("pool").set(pool);
    output->compute();
    delete output;
  }
    
  n.clear();
  essentia::shutdown();

  return 0;
}
