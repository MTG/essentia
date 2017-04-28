/*
 * Copyright (C) 2006-2017  Music Technology Group - Universitat Pompeu Fabra
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

#include "chromaprintgenerator.h"

using namespace std;

namespace essentia {
namespace standard {

const char* ChromaprintGenerator::name = "Chromaprint";
const char* ChromaprintGenerator::category = "Fingerprinting";
const char* ChromaprintGenerator::description = DOC("");

void ChromaprintGenerator::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _maxLength = parameter("maxLength").toReal();
}

void ChromaprintGenerator::compute() {
  const std::vector<Real>& signal = _signal.get();
  std::string& fingerprint = _fingerprint.get();
  char *fp;
  int inputSize;

  _maxLength == 0. ? inputSize = signal.size() : inputSize = _sampleRate * _maxLength;

  if (inputSize <= 0) {
    throw EssentiaException("Chromaprinter: the number of samples to compute Chromaprint should be grater than 0 but it is ", inputSize);
  }

  // Copy the signal to new vector to expand it to the int16_t dynamic range before the cast.
  std::vector<Real> signalScaled = signal;
  std::transform(signalScaled.begin(), signalScaled.end(), signalScaled.begin(),
                 std::bind1st(std::multiplies<Real>(), pow(2,15)));

  std::vector<int16_t> signalCast(signalScaled.begin(), signalScaled.end());

  const int num_channels = 1;

  _ctx = chromaprint_new(CHROMAPRINT_ALGORITHM_DEFAULT);

  chromaprint_start(_ctx, (int)_sampleRate, num_channels);

  chromaprint_feed(_ctx, &signalCast[0], inputSize);

  chromaprint_finish(_ctx);

  chromaprint_get_fingerprint(_ctx, &fp);

  fingerprint = const_cast<char*>(fp);

  chromaprint_dealloc(fp);

  chromaprint_free(_ctx);

/*  @endcode

  Note that there is no error handling in the code above. Almost any of the called functions can fail.
  You should check the return values in an actual code.
 */
}

} // namespace standard
} // namespace essentia
