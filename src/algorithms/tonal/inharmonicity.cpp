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

#include "inharmonicity.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Inharmonicity::name = "Inharmonicity";
const char* Inharmonicity::description = DOC("This algorithm calculates the inharmonicity of a signal given its spectral peaks. The inharmonicity value is computed as an energy weighted divergence of the spectral components from their closest multiple of the fundamental frequency. The fundamental frequency is taken as the first spectral peak from the input. The inharmonicity value ranges from 0 (purely harmonic signal) to 1 (inharmonic signal).\n\n"
"Inharmonicity was designed to be fed by the output from the HarmonicPeaks algorithm. Note that DC components should be removed from the signal before obtaining its peaks. An exception is thrown if a peak is given at 0Hz.\n\n"
"An exception is thrown if frequency vector is not sorted in ascendently, if it contains duplicates or if any input vector is empty.\n\n"
"References:\n"
"  [1] G. Peeters, \"A large set of audio features for sound description\n"
"  (similarity and classification) in the CUIDADO project,\" CUIDADO I.S.T.\n"
"  Project Report, 2004.\n\n"
"  [2] Inharmonicity - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Inharmonicity");


void Inharmonicity::compute() {

  const vector<Real>& frequencies = _frequencies.get();
  const vector<Real>& magnitudes = _magnitudes.get();
  Real& inharmonicity = _inharmonicity.get();

  if (magnitudes.size() != frequencies.size()) {
    throw EssentiaException("Inharmonicity: frequency and magnitude vectors have different size");
  }

  if (frequencies.empty()) {
    inharmonicity = 0.0;
    return;
    //throw EssentiaException("Inharmonicity: input vectors empty");
  }

  Real f0 = frequencies[0];
  if (f0 == 0) {
    throw EssentiaException("Inharmonicity: fundamental frequency found at 0 Hz");
  }

  Real ratio = 1.0;
  Real num = 0.0;
  Real den = magnitudes[0] * magnitudes[0];

  for (int i=1; i<int(frequencies.size()); ++i) {
    // validate input
    if (frequencies[i] < frequencies[i-1]) {
       throw EssentiaException("Inharmonicity: spectral peaks must be sorted in ascending-frequency order");
    }
    if (frequencies[i] == frequencies[i-1]) {
       throw EssentiaException("Inharmonicity: duplicate spectral peak frequency cannot exist");
    }

    // first find what will be the closest harmonic:
    ratio = round(frequencies[i]/f0);
    num += abs(frequencies[i] - ratio * f0) * magnitudes[i] * magnitudes[i];
    den += magnitudes[i] * magnitudes[i];
  }

  if (den == 0.0) {
    inharmonicity = 1.0;
  }
  else {
    inharmonicity = num/(den*f0);
  }
}
