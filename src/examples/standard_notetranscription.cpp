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
  int hopsize = 128;
  int sr = 44100;


  // instanciate facgory and create algorithms:
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* audioload = factory.create("MonoLoader",
                                       "filename", argv[1],
                                       "sampleRate", sr,
                                       "downmix", "mix");

  Algorithm* pitchDetect = factory.create("PitchMelodia",
                                          "sampleRate", sr, "hopSize", hopsize, "frameSize", framesize);
    
  Algorithm* noteSeg = factory.create("PitchContourSegmentation", "sampleRate", sr, "hopSize", hopsize);
    

  // data storage
  Pool pool;

  // set audio load:
  vector<Real> audio;
  audioload->output("audio").set(audio);
    
  // set pitch detection
  vector<Real> pitch, pitchConfidence;
  pitchDetect->input("signal").set(audio);
  pitchDetect->output("pitch").set(pitch);
  pitchDetect->output("pitchConfidence").set(pitchConfidence);
  
  // set note segmentation
  vector<Real> onset, duration, MIDIpitch;
  noteSeg->input("pitch").set(pitch);
  noteSeg->input("signal").set(audio);
  noteSeg->output("onset").set(onset);
  noteSeg->output("duration").set(duration);
  noteSeg->output("MIDIpitch").set(MIDIpitch);
    
  // load audio:
  audioload->compute();

  // extract monophonic melody using the MELODIA algorithm
  pitchDetect->compute();
    
  // segment the resulting contour into discrete note events
  noteSeg->compute();
    
  pool.add( "tonal.onset", onset);
  pool.add( "tonal.duration", duration);
  pool.add( "tonal.MIDIpitch", MIDIpitch);
    
  if (argc == 4 && atoi(argv[3])) {
    // printing to stdout:
    cout << "onset      duration      MIDI pitch" << endl;
    for (int i=0; i<onset.size(); i++){
        cout << onset[i] << "   " << duration[i] << "   " << MIDIpitch[i] << endl;
    }
  }

  // write to yaml file:
  Algorithm* output = factory.create("YamlOutput", "filename", argv[2]);
  output->input("pool").set(pool);
  output->compute();

  // clean up:
  delete audioload;
  delete pitchDetect;

  essentia::shutdown();

  return 0;
}
