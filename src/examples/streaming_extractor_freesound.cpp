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

#ifdef _WIN32
#include <windows.h>
#endif

#include <essentia/essentia.h>
#include <essentia/algorithm.h>
#include <essentia/algorithmfactory.h> 
#include <essentia/utils/extractor_freesound/extractor_version.h>
#include "music_extractor/extractor_utils.h"

#include "credit_libav.h"

using namespace std;
using namespace essentia;
using namespace essentia::standard;

// this code pretty much doubles streaming_extractor_music

void usage(char *progname) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << progname << " input_audiofile output_textfile [profile]" << endl;
    cout << "See http://essentia.upf.edu/documentation/streaming_extractor_music.html" << endl;
    cout << endl << "Freesound extractor version '" << FREESOUND_EXTRACTOR_VERSION << "'" << endl 
         << "built with Essentia version " << essentia::version_git_sha << endl;
    creditLibAV();

    exit(1);
}


int essentia_main(string audioFilename, string outputFilename, string profileFilename) {
  // Returns: 1 on essentia error

  try {
    essentia::init();

    cout.precision(10); // TODO ????

    Pool options;
    setExtractorDefaultOptions(options);
    setExtractorOptions(profileFilename, options);

    Algorithm* extractor = AlgorithmFactory::create("FreesoundExtractor",
                                                    "profile", profileFilename);

    Pool results;
    Pool resultsFrames;

    extractor->input("filename").set(audioFilename);
    extractor->output("results").set(results);
    extractor->output("resultsFrames").set(resultsFrames);

    extractor->compute();

    mergeValues(results, options);

    outputToFile(results, outputFilename, options);    
    if (options.value<Real>("outputFrames")) {
      outputToFile(resultsFrames, outputFilename+"_frames", options);
    }
    delete extractor;
    essentia::shutdown();
  }
  catch (EssentiaException& e) {
    cerr << e.what() << endl;
    return 1;
  }
  catch (const std::bad_alloc& e) {
    cerr << "bad_alloc exception: Out of memory " << e.what() << endl;
    return 1;
  }

  return 0;
}

#ifdef _WIN32
int main(int win32_argc, char **win32_argv)
{
  int i, argc = 0, buffsize = 0, offset = 0;
  char **utf8_argv, *utf8_argv_ptr;
  wchar_t **argv;

  argv = CommandLineToArgvW(GetCommandLineW(), &argc);

  buffsize = 0;
  for (i = 0; i < argc; i++) {
      buffsize += WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, NULL, 0, NULL, NULL);
  }

  size_t len = sizeof(char *) * (argc + 1) + buffsize;
  utf8_argv = (char**)malloc(len);
  memset(utf8_argv, 0, len);
  utf8_argv_ptr = (char *)utf8_argv + sizeof(char *) * (argc + 1);

  for (i = 0; i < argc; i++) {
      utf8_argv[i] = &utf8_argv_ptr[offset];
      offset += WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, &utf8_argv_ptr[offset], buffsize - offset, NULL, NULL);
  }

  LocalFree(argv);

  string audioFilename, outputFilename, profileFilename;

  switch (argc) {
    case 3:
      audioFilename =  utf8_argv[1];
      outputFilename = utf8_argv[2];
      break;
    case 4: // profile supplied
      audioFilename =  utf8_argv[1];
      outputFilename = utf8_argv[2];
      profileFilename = utf8_argv[3];
      break;
    default:
      usage(utf8_argv[0]);
  }

  return essentia_main(audioFilename, outputFilename, profileFilename);
}

#else
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

  return essentia_main(audioFilename, outputFilename, profileFilename);
}
#endif