/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "flatnessdb.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* FlatnessDB::name = "FlatnessDB";
const char* FlatnessDB::description = DOC("This algorithm computes the flatness of an array, which is defined as the ratio between the geometric mean and the arithmetic mean and it converts it to dB scale."
"\n"
"The size of the input array must be greater than 0. If the input array is empty an exception will be thrown. This algorithm uses the Flatness algorithm and thus inherits its input requirements and exceptions.\n"
"\n"
"References:\n"
"  [1] G. Peeters, A large set of audio features for sound description (similarity and classification) in the CUIDADO project,"
"      CUIDADO I.S.T. Project Report, 2004");

void FlatnessDB::compute() {

  const std::vector<Real>& array = _array.get();

  if (array.empty()) {
    throw EssentiaException("FlatnessDB: size of input array is zero");
  }

  Real& flatnessDB = _flatnessDB.get();

  Real flatness;

  _flatness->input("array").set(array);
  _flatness->output("flatness").set(flatness);
  _flatness->compute();

  if (flatness <= 0.0) {
    flatnessDB = 1.0; // default value chosen for silent signals
  }
  else {
    flatnessDB = std::min( Real(lin2db(flatness)/-60.0), Real(1.0));
  }
}
