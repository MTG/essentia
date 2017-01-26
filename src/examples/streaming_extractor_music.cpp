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

// Streaming extractor designed for analysis of music collections

#ifdef _WIN32
#include <windows.h>
#endif

#include <essentia/essentia.h>
#include <essentia/algorithm.h>
#include <essentia/algorithmfactory.h> 
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/essentiautil.h>
#include <essentia/utils/extractor_music/extractor_version.h>

#include "credit_libav.h" 


using namespace std;
using namespace essentia;
using namespace essentia::standard;
//using namespace essentia::streaming;
//using namespace essentia::scheduler;


void usage(char *progname) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << progname << " input_audiofile output_textfile [profile]" << endl;
    cout << endl << "Music extractor version '" << EXTRACTOR_VERSION << "'" << endl 
         << "built with Essentia version " << essentia::version_git_sha << endl;
    creditLibAV();

    exit(1);
}


void setExtractorDefaultOptions(Pool &options) {
  options.set("outputFrames", false);
  options.set("outputFormat", "json");
  options.set("requireMbid", false);
  options.set("indent", 4);

  options.set("highlevel.inputFormat", "json");
}


void setExtractorOptions(const std::string& filename, Pool& options) {
  setExtractorDefaultOptions(options);
  if (filename.empty()) return;

  Pool opts;
  Algorithm * yaml = AlgorithmFactory::create("YamlInput", "filename", filename);
  yaml->output("pool").set(opts);
  yaml->compute();
  delete yaml;
  options.merge(opts, "replace");
}


void outputToFile(Pool& pool, const string& outputFilename, Pool& options) {

  cerr << "Writing results to file " << outputFilename << endl;
  int indent = (int)options.value<Real>("indent");

  string format = options.value<string>("outputFormat");
  Algorithm* output = AlgorithmFactory::create("YamlOutput",
                                               "filename", outputFilename,
                                               "doubleCheck", true,
                                               "format", format,
                                               "writeVersion", false,
                                               "indent", indent);
  output->input("pool").set(pool);
  output->compute();
  delete output;
}


int essentia_main(string audioFilename, string outputFilename, string profileFilename) {
  // Returns: 1 on essentia error
  //          2 if there are no tags in the file
  // TODO: is 2 recieved?

  int result;
  try {
    essentia::init();

    cout.precision(10); // TODO ????

    Pool options;
    setExtractorOptions(profileFilename, options);

    Algorithm* extractor = AlgorithmFactory::create("MusicExtractor",
                                                    "profile", profileFilename);
    /*
    MusicExtractor *extractor = new MusicExtractor();

    extractor->setExtractorOptions(profileFilename);
    extractor->mergeValues(extractor->results);
    */

    Pool results;
    Pool resultsFrames;

    extractor->input("filename").set(audioFilename);
    extractor->output("results").set(results);
    extractor->output("resultsFrames").set(resultsFrames);

    extractor->compute();  

    outputToFile(results, outputFilename, options);
    
    if (options.value<Real>("outputFrames")) {
      outputToFile(resultsFrames, outputFilename+"_frames", options);
    }

    essentia::shutdown();
  }
  catch (EssentiaException& e) {
    cerr << e.what() << endl;
    return 1;
  }
  return result;

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
