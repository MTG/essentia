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

#include "median.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* Median::name = "Median";
const char* Median::description = DOC("This algorithm computes the median of an array of Reals. When there is an odd number of numbers, the median is simply the middle number. For example, the median of 2, 4, and 7 is 4. When there is an even number of numbers, the median is the mean of the two middle numbers. Thus, the median of the numbers 2, 4, 7, 12 is (4+7)/2 = 5.5. See [1] for more info.\n"
"\n"
"References:\n"
"  [1] Statistical Median -- from Wolfram MathWorld,\n"
"  http://mathworld.wolfram.com/StatisticalMedian.html");

void Median::compute() {
  _median.get() = median(_array.get());
}
