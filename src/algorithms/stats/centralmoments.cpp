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

#include "centralmoments.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* CentralMoments::name = "CentralMoments";
const char* CentralMoments::category = "Statistics";
const char* CentralMoments::description = DOC("This algorithm extracts the 0th, 1st, 2nd, 3rd and 4th central moments of an array. It returns a 5-tuple in which the index corresponds to the order of the moment.\n"
"\n"
"Central moments cannot be computed on arrays which size is less than 2, in which case an exception is thrown.\n"
"\n"
"Note: the 'mode' parameter defines whether to treat array values as a probability distribution function (pdf) or as sample points of a distribution (sample).\n"
"\n"
"References:\n"
"  [1] Sample Central Moment -- from Wolfram MathWorld,\n"
"  http://mathworld.wolfram.com/SampleCentralMoment.html\n\n"
"  [2] Central Moment - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Central_moment");

void CentralMoments::configure() {
  _mode = parameter("mode").toLower();
  _range = parameter("range").toReal();
}

void CentralMoments::compute() {

  const std::vector<Real>& array = _array.get();
  std::vector<Real>& centralMoments = _centralMoments.get();
  centralMoments.resize(5);

  if (array.empty()) {
    throw EssentiaException("CentralMoments: cannot compute the central moments of an empty array");
  }

  if (array.size() == 1) {
    throw EssentiaException("CentralMoments: cannot compute the central moments of an array of size 1");
  }

  if (_mode == "sample") {
    // treat array values as a sample of distribution

    // compute mean
    double m = 0.;
    for (int i=0; i<(int)array.size(); i++) {
      m += array[i]; 
    }
    m /= array.size();

    // compute central moments
    double sum2 = 0., sum3 = 0., sum4 = 0.;
    double x, x2;

    for (int i=0; i<(int)array.size(); i++) {
      x = array[i] - m;
      x2 = x * x;
      sum2 += x2;
      sum3 += x2 * x;
      sum4 += x2 * x2;
    }

    centralMoments[0] = 1.;
    centralMoments[1] = 0.;
    centralMoments[2] = sum2 / array.size();
    centralMoments[3] = sum3 / array.size();
    centralMoments[4] = sum4 / array.size();
  }
  else if (_mode == "pdf") {
    // treat array values as a probability density function

    // For precision reasons, we first compute the central moments with a normalized
    // range [0,1], and we multiply by the desired range at the end only.

    // scale is the horizontal scale, thus i*scale corresponds to the
    // normalized frequency, i.e.: between 0 and 1
    double scale = (double)1.0 / (array.size() - 1);

    double norm = 0.0;
    for (int i=0; i<(int)array.size(); i++) norm += array[i];

    if (norm == 0.0) {
      for (int k=0; k<5; k++) {
        centralMoments[k] = 0.0;
      }
      return;
    }

    // centroid is also in normalized frequency, i.e.: between 0 and 1
    double centroid = 0.0;
    for (int i=0; i<(int)array.size(); i++) {
      centroid += (i*scale) * array[i];
    }
    centroid /= norm;

    centralMoments[0] = 1.0;
    centralMoments[1] = 0.0;

    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    double v, v2, v2f;

    for (int i=0; i<(int)array.size(); i++) {
      v = (i*scale) - centroid;
      v2 = v*v;
      v2f = v2 * array[i];
      m2 += v2f;
      m3 += v2f * v;
      m4 += v2f * v2;
    }

    m2 /= norm;
    m3 /= norm;
    m4 /= norm;

    // we want the results inside the specified range, so as we factored it out
    // in the above formula, we have to inject it again to get back the results
    // relative to the desired range.
    double r = _range;
    centralMoments[2] = m2 * r*r;
    centralMoments[3] = m3 * r*r*r;
    centralMoments[4] = m4 * r*r*r*r;
  } 
}
