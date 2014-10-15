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

// Streaming extractor designed for analysis of music collections

#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/essentiautil.h>

#include "extractor_music/MusicExtractor.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

void usage() {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: streaming_extractor_archivemusic input_audiofile output_textfile [profile]" << endl;
    exit(1);
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
      usage();
  }

  try {
    essentia::init();

    cout.precision(10); // TODO ????
  
    MusicExtractor *extractor = new MusicExtractor();
    
    extractor->setExtractorOptions(profileFilename);
    extractor->mergeValues();

    extractor->compute(audioFilename);

    extractor->outputToFile(extractor->stats, outputFilename);
    if (extractor->options.value<Real>("outputFrames")) { 
      extractor->outputToFile(extractor->results, outputFilename+"_frames");
    }
      
    essentia::shutdown();
  }
  catch (EssentiaException& e) {
    cout << e.what() << endl;
    return 1;
  }
  return 0;
}