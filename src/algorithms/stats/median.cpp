/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
"      http://mathworld.wolfram.com/StatisticalMedian.html");

void Median::compute() {
  _median.get() = median(_array.get());
}
