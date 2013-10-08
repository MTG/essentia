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

#include "essentia_gtest.h"
#include "network.h"
#include "networkparser.h"
#include "graphutils.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;


inline string getName(NetworkNode* node) {
  return node->algorithm()->name();
}

inline Algorithm* getAlgo(NetworkNode* node) {
  return node->algorithm();
}

/**
 * Test that scheduling works when a source (from an algo in a composite) is
 * connected both to a SourceProxy and another inner algorithm.
 * eg: for barkbands, as in LowLevelSpectralExtractor, etc...
 */
TEST(Scheduler, SourceProxyFork) {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* loader = factory.create("MonoLoader",
                                     "filename", "test/audio/recorded/cat_purrrr.wav");

  Algorithm* bbands = factory.create("BarkExtractor");

  loader->output("audio")    >>  bbands->input("signal");

  bbands->output("barkbands")             >>  NOWHERE;
  bbands->output("barkbands_kurtosis")    >>  NOWHERE;
  bbands->output("barkbands_skewness")    >>  NOWHERE;
  bbands->output("barkbands_spread")      >>  NOWHERE;
  bbands->output("spectral_crest")        >>  NOWHERE;
  bbands->output("spectral_flatness_db")  >>  NOWHERE;

  Network network(loader);

  network.run();
}
