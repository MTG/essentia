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

#include "distributionshape.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* DistributionShape::name = "DistributionShape";
const char* DistributionShape::description = DOC("This algorithm extracts the spread (variance), skewness and kurtosis excess of an array of Reals given its central moments. These extracted features are good indicators of the shape of the distribution.\n"
"The size of the input array must be at least 5. An exception will be thrown otherwise.\n"
"\n"
"References:\n"
"  [1] G. Peeters, \"A large set of audio features for sound description\n"
"  (similarity and classification) in the CUIDADO project,\" CUIDADO I.S.T.\n"
"  Project Report, 2004.\n\n"
"  [2] Variance - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Variance\n\n"
"  [3] Skewness - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Skewness\n\n"
"  [4] Kurtosis - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Kurtosis");


void DistributionShape::compute() {

  const std::vector<Real>& centralMoments = _centralMoments.get();
  Real& spread = _spread.get();
  Real& skewness = _skewness.get();
  Real& kurtosis = _kurtosis.get();

  if (centralMoments.size() != 5) {
    throw EssentiaException("DistributionShape: the size of 'centralMoments' input is not 5");
  }

  spread = centralMoments[2];

  if (spread == 0.0) skewness = 0.0;
  else skewness = (Real)(centralMoments[3] / pow(spread, (Real)1.5));

  if (spread == 0.0) kurtosis = -3.0;
  else kurtosis = (centralMoments[4] / (spread * spread)) - 3.0;
}
