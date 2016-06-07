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
#include <essentia/streaming/algorithms/poolstorage.h>
#include <essentia/scheduler/network.h>
#include "credit_libav.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

int main(int argc, char* argv[]) {
  essentia::init();
  Pool pool;

  cout << "MD5 extractor computes MD5 value over undecoded audio payload of a file ignoring metadata. It can be used to identify duplicates, that is, files that have the same audio content although their metadata can differ." << endl;

  if (argc != 2) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile" << endl;
    creditLibAV();
    exit(1);
  }
  string audioFilename = argv[1];

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();
  Algorithm* loader = factory.create("AudioLoader",
                                     "filename", audioFilename,
                                     "computeMD5", true);
  loader->output("audio") >> NOWHERE;
  loader->output("sampleRate") >> NOWHERE;
  loader->output("numberChannels") >> NOWHERE;
  loader->output("codec") >> NOWHERE;
  loader->output("bit_rate") >> NOWHERE;
  loader->output("md5") >> PC(pool, "md5");

  Network network(loader);
  network.run();

  cout << "MD5: " << pool.value<string>("md5") << endl;
  essentia::shutdown();

  return 0;
}
