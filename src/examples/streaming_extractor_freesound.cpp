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

#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/essentiautil.h>
#include <essentia/essentia.h> 

#include "freesound/FreesoundExtractor.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;


void usage() {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: streaming_extractor input_audiofile output_textfile [profile]" << endl;
    exit(1);
}


int main(int argc, char* argv[]) {

  string audioFilename, outputFilename, profileFilename;
  Pool options,results;

  switch (argc) {
    case 3:
      audioFilename =  argv[1];
      outputFilename = argv[2];
      break;
    default:
      usage();
  }
  
  try {
    essentia::init();
  
    FreesoundExtractor *extractor = new FreesoundExtractor();
    extractor->compute(audioFilename);
    extractor->outputToFile(extractor->stats, outputFilename+".json", true);
    extractor->outputToFile(extractor->stats, outputFilename+".yaml", false);
    extractor->outputToFile(extractor->results, outputFilename+"_frames.json", true);
  
    essentia::shutdown();
  }
  catch (EssentiaException& e) {
    cout << e.what() << endl;
    return 1;
  }
  return 0;
}