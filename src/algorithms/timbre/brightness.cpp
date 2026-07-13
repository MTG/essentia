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
const char* Brightness::description = DOC("TODO");


void Brightness::configure() {
}

void Brightness::compute() {

  const vector<Real>& signal = _signal.get();
  Real& brightness = _brightness.get();
  brightness = 50.0;
}
