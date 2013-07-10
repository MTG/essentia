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

/*detects onsets and save them into a file.
 by Piotr Holonowicz, MTG, UPF
*/
#include <iostream>
#include <fstream>
#include "algorithmfactory.h"
using namespace std;
using namespace essentia;
using namespace standard;

int save_onsets(const std::string& outputName,const vector<Real>& onsets)
{
  try
  {
    std::fstream dataFile(outputName.c_str(),std::ios_base::out);
    if(!dataFile.is_open()) { throw("Could not open output file!"); }
    {
      vector<Real>::const_iterator iter;
      for(iter = onsets.begin();iter != onsets.end();iter++)
      {
        float val = *iter;
        dataFile << val << std::endl;
      }
    }
    dataFile.close();
  }
  catch(const char* text)
  {
    cout << "Fatal error : " << text << ", exiting... " << std::endl;
    return EXIT_FAILURE;
  }
  catch(std::exception& e)
  {
    cout << e.what() << "" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


int main(int argc, char* argv[])
{
  cout << "Essentia onset detector (weighted Complex and HFC detection functions)" << endl;
  if (argc != 2) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile" << endl;
    exit(1);
  }

  essentia::init();

  Real onsetRate;
  vector<Real> onsets;

  vector<Real> audio, unused;
  // cut the file extension to create output file later
  std::string fileName = argv[1];
  size_t pos = fileName.find(".wav");
  std::string fileNoExt = fileName.substr(0,pos); // take the first pos characters
  cout << "The input file name without extension is : " << fileNoExt << "" << std::endl;

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
  save_onsets(fileNoExt+".onset_computed",onsets);
  delete extractoronsetrate;
  delete audiofile;

  essentia::shutdown();

  return 0;
}
