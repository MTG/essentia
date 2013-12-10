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

#ifndef FREESOUND_EXTRACTOR_H
#define FREESOUND_EXTRACTOR_H

#include  "essentia/pool.h"
#include  "essentia/algorithm.h"
#include  "essentia/types.h"
#include  "essentia/scheduler/network.h"
#include  "essentia/streaming/sourcebase.h"
#include  "essentia/streaming/streamingalgorithm.h"
#include  "essentia/algorithmfactory.h"
#include  "essentia/streaming/algorithms/poolstorage.h"
 #include "essentia/streaming/algorithms/vectorinput.h"

#include "FreesoundLowlevelDescriptors.h"
#include "FreesoundRhythmDescriptors.h"
#include "FreesoundTonalDescriptors.h"
#include "FreesoundSfxDescriptors.h"

#define EXTRACTOR_VERSION "0.2"

using namespace std;
using namespace essentia;
using namespace streaming;

 class FreesoundExtractor{

 protected:

 	Pool computeAggregation(Pool& pool);

 public:
 	Pool results;
 	Pool stats;

 	void compute(const string& audioFilename);

	void outputToFile(Pool& pool, const string& outputFilename, bool outputJSON);
	
 };

 #endif