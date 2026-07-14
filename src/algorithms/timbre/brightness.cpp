/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#include "brightness.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Brightness::name = "Brightness";
const char* Brightness::category = "Timbre";
const char* Brightness::description = DOC("This algorithm computes the brightness of the analyzed audio in a scale from [0-100]. A bright sound is one that is clear/vibrant and/or contains significant high-pitched elements.\n"
"\n"
"References:\n"
"  [1] A. Pearce, S. Safavi, T. Brookes, R. Mason, W. Wang, and M. Plumbley, \"AudioCommons Timbral Models,\" Github Repository, "
"  https://github.com/AudioCommons/timbral_models.\n\n"
"  [2] A. Pearce, S. Safavi, T. Brookes, R. Mason, W. Wang, and M. Plumbley, \"D5.8: Release of timbral characterisation "
"  tools for semantically annotating non-musical content\", January 2019. https://audiocommons.github.io/materials/.\n");


void Brightness::configure() {
}

void Brightness::compute() {

  const vector<Real>& signal = _signal.get();
  
  const Real energyThreshold = parameter("energyThreshold").toReal();
  const Real ratioCrossover = parameter("ratioCrossover").toReal();
  const Real centroidCrossover = parameter("centroidCrossover").toReal();
  const int hopSize = parameter("hopSize").toInt();
  const int windowSize = parameter("windowSize").toInt();
  const Real minFreq = parameter("minFreq").toReal();


  Real& brightness = _brightness.get();
  brightness = 50.0;
}
