/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "distributionshape.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* DistributionShape::name = "DistributionShape";
const char* DistributionShape::description = DOC("This algorithm extracts the spread (variance), skewness and kurtosis excess of an array of Reals given its central moments. These extracted features are good indicators of the shape of the distribution.\n"
"The size of the input array must be at least 5. An exception will be thrown otherwise.\n"
"References:\n"
"  [1] G. Peeters, A large set of audio features for sound description (similarity and classification) in the CUIDADO project,\n"
"      CUIDADO I.S.T. Project Report, 2004\n"
"  [2] Variance - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Variance\n"
"  [3] Skewness - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Skewness\n"
"  [4] Kurtosis - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Kurtosis");


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
