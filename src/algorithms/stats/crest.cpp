/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "crest.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Crest::name = "Crest";
const char* Crest::description = DOC("This algorithm computes the crest of an array. The crest is defined as the ratio between the maximum value and the arithmetic mean of an array. Typically it is used on the magnitude spectrum.\n"
"\n"
"Crest cannot be computed neither on empty arrays nor arrays which contain negative values. In such cases, exceptions will be thrown.\n"
"\n"
"References:\n"
"  [1] G. Peeters, A large set of audio features for sound description \n"
"      (similarity and classification) in the CUIDADO project, \n"
"      CUIDADO I.S.T. Project Report, 2004");

void Crest::compute() {

  const vector<Real>& array = _array.get();
  Real& crest = _crest.get();

  if (array.size() == 0) {
    throw EssentiaException("Crest: array does not contain any values");
  }

  if (find_if(array.begin(), array.end(), bind2nd(less<Real>(), 0)) != array.end()) {
    throw EssentiaException("Crest: array must not contain negative values");
  }

  Real maximum = *max_element(array.begin(), array.end());

  if (maximum == 0.0) {
    crest = 0.0;
  }
  else {
    Real arithmeticMean = mean(array);
    crest = maximum / arithmeticMean;
  }
}
