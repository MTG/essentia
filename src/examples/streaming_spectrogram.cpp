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

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

int main(int argc, char* argv[]) {

  if (argc != 3) {
    cout << "Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input json_output" << endl;
    cout << "Computes spectrogram and mel-band log-energies spectrogram for the input audiofile." << endl;
    creditLibAV();
    exit(1);
  }

  string audioFile = argv[1];
  string outputFile = argv[2];
  string outputSpecFile = outputFile + ".spec";

  essentia::init();

  Pool pool;

  // TODO this should be configurable
  Real sampleRate = 44100.0;
  int frameSize = 2048;
  int hopSize = 1024;
  std::string format ("json");

  AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFile,
                                    "sampleRate", sampleRate);

  Algorithm* fc    = factory.create("FrameCutter",
                                    "frameSize", frameSize,
                                    "hopSize", hopSize,
                                    "silentFrames", "noise");

  Algorithm* w     = factory.create("Windowing",
                                    "type", "blackmanharris62");

  Algorithm* spec  = factory.create("Spectrum");
  Algorithm* mfcc  = factory.create("MFCC", "numberBands", 40,
                                    "numberCoefficients", 20);
  Algorithm* melbands96 = factory.create("MelBands",
                                         "numberBands", 96,
                                         "log", true);

  Algorithm* file = new FileOutput<vector<Real> >();
  file->configure("filename", outputSpecFile, "mode", "binary");

  // Audio -> FrameCutter -> Windowing -> Spectrum
  audio->output("audio") >> fc->input("signal");
  fc->output("frame") >> w->input("frame");
  w->output("frame") >> spec->input("frame");

  // Spectrum -> MFCC -> Pool
  spec->output("spectrum") >> mfcc->input("spectrum");
  spec->output("spectrum") >> melbands96->input("spectrum");
  spec->output("spectrum") >> file->input("data");

  mfcc->output("bands") >> NOWHERE; // only store high-res mel bands
  mfcc->output("mfcc") >> PC(pool, "lowlevel.mfcc");
  melbands96->output("bands") >> PC(pool, "lowlevel.melbands96");

  cout << "Analyzing " << audioFile << endl;;

  Network n(audio);
  n.run();

  // write results to file
  cout << "Writing results to json file " << outputFile << endl;
  cout << "Writing spectrogram to binary file " << outputSpecFile << endl;

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", outputFile,
                                                                   "format", format);

  output->input("pool").set(pool);
  output->compute();

  n.clear();

  delete output;
  essentia::shutdown();

  return 0;
}
