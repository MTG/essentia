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

#include "entropy.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Entropy::name = "Entropy";
const char* Entropy::category = "Statistics";
const char* Entropy::description = DOC("This algorithm computes the Shannon entropy of an array. Entropy can be used to quantify the peakiness of a distribution. This has been used for voiced/unvoiced decision in automatic speech recognition. \n"
"\n"
"Entropy cannot be computed neither on empty arrays nor arrays which contain negative values. In such cases, exceptions will be thrown.\n"
"\n"
"References:\n"
"  [1] H. Misra, S. Ikbal, H. Bourlard and H. Hermansky, \"Spectral entropy\n"
"  based feature for robust ASR,\" in IEEE International Conference on\n"
"  Acoustics, Speech, and Signal Processing (ICASSP'04).");

void Entropy::compute() {
    vector<Real> array = _array.get();
    Real& entropy = _entropy.get();
    
    if (array.size() == 0) {
        throw EssentiaException("Entropy: array does not contain any values");
    }
    
    if (find_if(array.begin(), array.end(), bind2nd(less<Real>(), 0)) != array.end()) {
        throw EssentiaException("Entropy: array must not contain negative values");
    }
    
    normalizeSum(array);
    entropy = 0.0;

    for (size_t i=0; i<array.size(); ++i) {
        if (array[i]==0)array[i] = 1;
        entropy -= log2(array[i]) * array[i];
    }
}
