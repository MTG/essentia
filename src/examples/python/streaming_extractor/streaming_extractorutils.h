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

#ifndef STREAMING_EXTRACTOR_UTILS_H
#define STREAMING_EXTRACTOR_UTILS_H

#include "algorithmfactory.h"
#include "pool.h"
#include "types.h"
#include <string>

void pcmMetadata(essentia::streaming::AlgorithmFactory& factory,
                 const std::string& audioFilename, essentia::Pool& pool);
void readMetadata(const std::string& audioFilename, essentia::Pool& pool);
void LevelAverage(essentia::Pool& pool, const std::string& nspace="");
void TuningSystemFeatures(essentia::Pool& pool, const std::string& nspace="");
void SFXPitch(essentia::Pool& pool, const std::string& nspace="");
void TonalPoolCleaning(essentia::Pool& pool, const std::string& nspace="");

void PCA(essentia::Pool& pool, const std::string& nspace="");
void PostProcess(essentia::Pool& pool, const std::string& nspace="");
void getAnalysisData(const essentia::Pool& pool, essentia::Real& replayGain,
                     essentia::Real& sampleRate, std::string& downmix);

#endif // STREAMING_EXTRACTOR_UTILS_H
