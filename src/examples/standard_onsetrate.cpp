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
using namespace std;
using namespace essentia;
using namespace standard;

int main(int argc, char* argv[]) {

  if (argc != 2) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile" << endl;
    exit(1);
  }

  essentia::init();

  Real onsetRate;
  vector<Real> onsets;

  vector<Real> audio, unused;

  // File Input
  Algorithm* audiofile = AlgorithmFactory::create("MonoLoader",
                                                  "filename", argv[1],
                                                  "sampleRate", 44100);

  Algorithm* extractoronsetrate = AlgorithmFactory::create("OnsetRate");

  audiofile->output("audio").set(audio);

  extractoronsetrate->input("signal").set(audio);
  extractoronsetrate->output("onsets").set(onsets);
  extractoronsetrate->output("onsetRate").set(onsetRate);

  audiofile->compute();
  extractoronsetrate->compute();

  cout << "onsetRate: " << onsetRate << endl;
  cout << "onsetTimes: " << onsets << endl;

  delete extractoronsetrate;
  delete audiofile;

  essentia::shutdown();

  return 0;
}
