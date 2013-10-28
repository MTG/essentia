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

#ifndef FREESOUND_LOWLEVEL_DESCRIPTORS_H
#define FREESOUND_LOWLEVEL_DESCRIPTORS_H

#include "FreesoundDescriptorsSet.h"
#include "essentia/essentiamath.h"

using namespace std;

 class FreesoundLowlevelDescriptors : public FreesoundDescriptorSet{

 public:
 	static const string nameSpace;  

 	void createNetwork(SourceBase& source, Pool& pool);
	void computeAverageLoudness(Pool& pool);

 };

 #endif