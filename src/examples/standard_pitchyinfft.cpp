/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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
#include <essentia/essentiamath.h> // for the isSilent function
#include <essentia/pool.h>

using namespace std;
using namespace essentia;
using namespace standard;


int main(int argc, char* argv[]) {

  if (argc < 3) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile output_yamlfile [0/1 to print to stdout]" << endl;
    exit(1);
  }

  // register the algorithms in the factory(ies)
  essentia::init();

  /////// PARAMS //////////////
  int framesize = 1024;
  int hopsize = 256;
  int sr = 44100;
  int zeropadding = 0;


  // instanciate facgory and create algorithms:
  AlgorithmFactory& factory = AlgorithmFactory::instance();

 Algorithm* audioload = factory.create("MonoLoader",
                                       "filename", argv[1],
                                       "sampleRate", sr,
                                       "downmix", "mix");

  Algorithm* frameCutter = factory.create("FrameCutter",
                                           "frameSize", framesize,
                                           "hopSize", hopsize,
                                           "startFromZero", false);

  Algorithm* window = factory.create("Windowing",
                                     "type", "hann",
                                     "zeroPadding", zeropadding);

  Algorithm* spectrum = factory.create("Spectrum",
                                       "size", framesize);

  Algorithm* pitchDetect = factory.create("PitchYinFFT",
                                          "frameSize", framesize,
                                          "sampleRate", sr);

  // data storage
  Pool pool;

  // set audio load:
  vector<Real> audio;
  audioload->output("audio").set(audio);

  // set frameCutter:
  vector<Real> frame;
  frameCutter->input("signal").set(audio);
  frameCutter->output("frame").set(frame);

  // set windowing:
  vector<Real> windowedframe;
  window->input("frame").set(frame);
  window->output("frame").set(windowedframe);

  // set spectrum:
  vector<Real> spec;
  spectrum->input("frame").set(windowedframe);
  spectrum->output("spectrum").set(spec);

  // set pitch extraction:
  Real thisPitch = 0., thisConf = 0;
  pitchDetect->input("spectrum").set(spec);
  pitchDetect->output("pitch").set(thisPitch);
  pitchDetect->output("pitchConfidence").set(thisConf);

  // load audio:
  audioload->compute();

  // process:
  while (true) {
    frameCutter->compute();

    if (!frame.size())
      break;

    if (isSilent(frame))
      continue;

    window->compute();
    spectrum->compute();
    pitchDetect->compute();

    pool.add( "tonal.pitch", thisPitch);
    pool.add( "tonal.pitch_confidence", thisConf);
  }

  if (argc == 4 && atoi(argv[3])) {
    // printing to stdout:
    const vector<Real>& pitches = pool.value<vector<Real> >("tonal.pitch");
    cout << "number of frames: " << pool.value<vector<Real> >("tonal.pitch").size() << endl;
    cout << pitches << endl;
  }

  // write to yaml file:
  Algorithm* output = factory.create("YamlOutput", "filename", argv[2]);
  output->input("pool").set(pool);
  output->compute();

  // clean up:
  delete audioload;
  delete frameCutter;
  delete spectrum;
  delete pitchDetect;

  essentia::shutdown();

  return 0;
}
