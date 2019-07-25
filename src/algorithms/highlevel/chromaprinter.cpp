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
const char* Chromaprinter::description = DOC("This algorithm computes the fingerprint of the input signal using Chromaprint algorithm. It is a wrapper of the Chromaprint library [1,2]. The chromaprints are returned as base64-encoded strings.\n"
"\n"
"References:\n"
"  [1] Chromaprint -- from the Acousticid project,\n"
"  https://acoustid.org/chromaprint\n\n"
"  [2] https://github.com/acoustid/chromaprint");

void Chromaprinter::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _maxLength = parameter("maxLength").toReal();
}

void Chromaprinter::compute() {
  const std::vector<Real>& signal = _signal.get();
  std::string& fingerprint = _fingerprint.get();
  unsigned inputSize;

  _maxLength == 0. ? inputSize = signal.size() : inputSize = _sampleRate * _maxLength;

  if (inputSize > signal.size()) inputSize = signal.size();

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

  char *fp;
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


namespace essentia {
namespace streaming {

const char* Chromaprinter::name = standard::Chromaprinter::name;
const char* Chromaprinter::description = standard::Chromaprinter::description;
const char* Chromaprinter::category = standard::Chromaprinter::category;


void Chromaprinter::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _analysisTime = parameter("analysisTime").toReal();
  _concatenate = parameter("concatenate").toBool();

  _chromaprintSize = _sampleRate * _analysisTime;

  _signal.setAcquireSize(4096);
  _signal.setReleaseSize(4096);

  _fingerprint.setAcquireSize(1);
  _fingerprint.setReleaseSize(1);

  _returnChromaprint = !_concatenate;

  _count = 0;
}

AlgorithmStatus Chromaprinter::process() {
  while (true) {
    EXEC_DEBUG("process()");

    AlgorithmStatus status = acquireData();

  /*
    EXEC_DEBUG("data acquired (in: " << _signal.acquireSize()
               << " - out: " << _outputAudio.acquireSize() << ")");
  */

    if (status != OK) {
      if (!shouldStop()) return status;

      // if shouldStop is true, that means there is no more audio coming, so we need
      // to take what's left to fill in half-frames, instead of waiting for more
      // data to come in (which would have done by returning from this function)

      int available = _signal.available();
      if (available == 0) return NO_INPUT;

      _signal.setAcquireSize(available);
      _signal.setReleaseSize(available);

      _chromaprintSize = _count + available;

      _returnChromaprint = true;

      return process();
    }

    const vector<Real>& signal = _signal.tokens();

    // Copy the signal to new vector to expand it to the int16_t dynamic range before the cast.
    std::vector<Real> signalScaled = signal;
    std::transform(signalScaled.begin(), signalScaled.end(), signalScaled.begin(),
                   std::bind1st(std::multiplies<Real>(), pow(2,15)));

    std::vector<int16_t> signalCast(signalScaled.begin(), signalScaled.end());

    if (_count == 0) initChromaprint();

    _buffer.insert(_buffer.end(), signalCast.begin(), signalCast.end());

    _count += signalCast.size();

    if (_count >= _chromaprintSize) {
      fingerprintConcatenated.append(getChromaprint());

      if (_returnChromaprint) {
        // Only acquires tokens when we want to output the chromaprint.
        _fingerprint.acquire(1);
        std::vector<std::string>& fingerprint = _fingerprint.tokens();
        fingerprint[0] = fingerprintConcatenated;
        _fingerprint.release();
        fingerprintConcatenated.erase();
      }

      _count = 0;
      _buffer.clear();
    }


    EXEC_DEBUG("releasing");
    _signal.release();
    EXEC_DEBUG("released");

    return OK;
  }
}


void Chromaprinter::initChromaprint(){
  const int num_channels = 1;
  _ctx = chromaprint_new(CHROMAPRINT_ALGORITHM_DEFAULT);

  _ok = chromaprint_start(_ctx, (int)_sampleRate, num_channels);
    if (!_ok) {
      throw EssentiaException("Chromaprinter: chromaprint_start returned error");
    }
}

std::string Chromaprinter::getChromaprint(){

  _ok = chromaprint_feed(_ctx, &_buffer[0], _chromaprintSize);
  if (!_ok) {
    throw EssentiaException("Chromaprinter: chromaprint_feed returned error");
  }

  _ok = chromaprint_finish(_ctx);
  if (!_ok) {
    throw EssentiaException("Chromaprinter: chromaprint_finish returned error");
  }

  char *fp;
  _ok = chromaprint_get_fingerprint(_ctx, &fp);
  if (!_ok) {
    throw EssentiaException("Chromaprinter: chromaprint_get_fingerprint returned error");
  }

  std::string chromaprint = const_cast<char*>(fp);

  chromaprint_dealloc(fp);

  chromaprint_free(_ctx);

  return chromaprint;
}

} // namespace streaming
} // namespace essentia
