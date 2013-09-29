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
#include <essentia/pool.h>
#include <essentia/utils/tnt/tnt2vector.h>

using namespace std;
using namespace essentia;
using namespace standard;

typedef TNT::Array2D<Real> array2d;

void AddToPool(const array2d& a2d,
               const string& desc, // descriptor names
               Pool& pool) {
    vector<vector<Real> > v2d =  array2DToVecvec(a2d);
    for (size_t i = 0; i < v2d.size(); ++i)
      pool.add(desc, v2d[i]);
}

int main(int argc, char** argv) {
  if (argc < 2) {
      cout << "Error: wrong number of arguments" << endl;
      cout << "Usage: " << argv[0] << " input_audiofile" << endl;
      exit(1);
  }

  essentia::init();

  // parameters:
  int sr = 44100;
  int framesize = sr/4;
  int hopsize = 256;
  Real frameRate = Real(sr)/Real(hopsize);

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", argv[1],
                                    "sampleRate", sr);

  Algorithm* frameCutter = factory.create("FrameCutter",
                                          "frameSize", framesize,
                                          "hopSize", hopsize);

  Algorithm* rms = factory.create("RMS");

  Algorithm* fadeDetect = factory.create("FadeDetection",
                                         "minLength", 3.,
                                         "cutoffHigh", 0.85,
                                         "cutoffLow", 0.20,
                                         "frameRate", frameRate);

  // create a pool for fades' storage:
  Pool pool;

  // set audio:
  vector<Real> audio_mono;
  audio->output("audio").set(audio_mono);

  // set frameCutter:
  vector<Real> frame;
  frameCutter->input("signal").set(audio_mono);
  frameCutter->output("frame").set(frame);

  // set rms:
  Real rms_value;
  rms->input("array").set(frame);
  rms->output("rms").set(rms_value);

  // we need a vector to store rms values:
  std::vector<Real> rms_vector;

  // load audio:
  audio->compute();

  // compute and store rms first and will compute fade detection later:
  while (true) {
    frameCutter->compute();
    if (frame.empty())
      break;

    rms->compute();
    rms_vector.push_back(rms_value);
  }

  // set fade detection:
  array2d fade_in;
  array2d fade_out;
  fadeDetect->input("rms").set(rms_vector);
  fadeDetect->output("fadeIn").set(fade_in);
  fadeDetect->output("fadeOut").set(fade_out);

  // compute fade detection:
  fadeDetect->compute();

  // Exemplifying how to add/retrieve values from the pool in order to output them  into stdout
  if (fade_in.dim1()) {
    AddToPool(fade_in, "high_level.fade_in", pool);
    vector<vector<Real> > fadeIn = pool.value<vector<vector<Real> > > ("high_level.fade_in");
    cout << "fade ins: ";
    for (size_t i=0; i < fadeIn.size(); ++i)
        cout << fadeIn[i] << endl;
  }
  else cout << "No fades in found" << endl;

  if (fade_out.dim1()) {
    AddToPool(fade_out, "high_level.fade_out", pool);
    vector<vector<Real> > fadeOut = pool.value<vector<vector<Real> > > ("high_level.fade_out");
    cout << "fade outs: ";
    for (size_t i=0; i < fadeOut.size(); ++i)
        cout << fadeOut[i] << endl;
  }
  else cout << "No fades out found" << endl ;

  delete audio;
  delete frameCutter;
  delete rms;
  delete fadeDetect;

  essentia::shutdown();

  return 0;
}
