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
#include <essentia/scheduler/network.h>
#include <essentia/streaming/algorithms/poolstorage.h>

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;


int main(int argc, char* argv[]) {

  if (argc < 3 ) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile output_yamlfile [1/0 print to stdout]" << endl;
    exit(1);
  }

  // register the algorithms in the factory(ies)
  essentia::init();

  /////// PARAMS //////////////
  int sr = 44100;


  // instanciate facgory and create algorithms:
  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audioload    = factory.create("MonoLoader",
                                           "filename", argv[1],
                                           "sampleRate", sr,
                                           "downmix", "mix");

  Algorithm* pitchDetect  = factory.create("PitchMelodia");
  Algorithm* vibrato      = factory.create("Vibrato");
  // by default, vibrato is adjusted to the predominant melody default output sample rate (44100.0/128.0)
    
  // data storage
  Pool pool;

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  // audio -> pitchDetect
  audioload->output("audio")              >>  pitchDetect->input("signal");
  // pitchDetect -> vibrato
  pitchDetect->output("pitch")            >>  vibrato->input("pitch");
  pitchDetect->output("pitch")            >>  PC(pool, "tonal.pitch");
  pitchDetect->output("pitchConfidence")  >>  NOWHERE;
  // vibrato -> pool
  vibrato->output("vibratoFrequency")  >>  PC(pool, "tonal.vibrato frequency");
  vibrato->output("vibratoExtend")  >>  PC(pool, "tonal.vibrato extend");
    

  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << argv[1]<< " --------" << endl;

  Network(audioload).run();

  // write results to yamlfile
  cout << "-------- writing results to file " << argv[2] << " --------" << endl;

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", argv[2]);

  output->input("pool").set(pool);

  output->compute();

  
  if (argc == 4 && atoi(argv[3])) {
    // printing to stdout:
    const vector<Real>& vibratoExtend = pool.value<vector<Real> >("tonal.vibrato extend");
    const vector<Real>& vibratoFrequency = pool.value<vector<Real> >("tonal.vibrato frequency");
    cout << "number of frames: " << vibratoExtend.size() << endl;
    for (int i=0; i<(int)vibratoExtend.size(); i++){
        cout << vibratoFrequency[i] << "   " << vibratoExtend[i]<< endl;
    }
  }
  
  // clean up:
  delete output;
  essentia::shutdown();

  return 0;
}
