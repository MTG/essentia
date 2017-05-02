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

#include "chromaprinter.h"

using namespace std;

namespace essentia {
namespace standard {

const char* Chromaprinter::name = "Chromaprinter";
const char* Chromaprinter::category = "Fingerprinting";
const char* Chromaprinter::description = DOC("");

void Chromaprinter::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _maxLength = parameter("maxLength").toReal();
}

void Chromaprinter::compute() {
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

  int ok;

  ok = chromaprint_start(_ctx, (int)_sampleRate, num_channels);
  if (!ok) {
    throw EssentiaException("Chromaprinter: chromaprint_start returned error");
  }

  ok = chromaprint_feed(_ctx, &signalCast[0], inputSize);
  if (!ok) {
    throw EssentiaException("Chromaprinter: chromaprint_feed returned error");
  }

  ok = chromaprint_finish(_ctx);
  if (!ok) {
    throw EssentiaException("Chromaprinter: chromaprint_finish returned error");
  }

  ok = chromaprint_get_fingerprint(_ctx, &fp);
  if (!ok) {
    throw EssentiaException("Chromaprinter: chromaprint_get_fingerprint returned error");
  }

  fingerprint = const_cast<char*>(fp);

  chromaprint_dealloc(fp);

  chromaprint_free(_ctx);


}

} // namespace standard
} // namespace essentia
