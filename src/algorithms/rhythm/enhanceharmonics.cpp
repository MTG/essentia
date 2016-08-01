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

#include "enhanceharmonics.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;

namespace essentia {
namespace standard {

const char* EnhanceHarmonics::name = "EnhanceHarmonics";
const char* EnhanceHarmonics::description = DOC("TODO add description\n");

void EnhanceHarmonics::configure() {
}

void EnhanceHarmonics::compute() {
  const vector<Real>& input = _input.get();
  vector<Real>& output = _output.get();

  /*
  Pseudocode:

  // TODO: boost harmonically related peaks
  output.reserve(...);
  for i in range(0, len(input)/4:
  	output.push(input[i] + input[2*i] + input[4*i]);
  */

  output = input;  
}

} // namespace standard
} // namespace essentia
