/*
 * Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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

#include "accumulatoralgorithm.h"
using namespace std;

namespace essentia {
namespace streaming {

AccumulatorAlgorithm::AccumulatorAlgorithm() : _preferredSize(0), _inputStream(0) {}

void AccumulatorAlgorithm::reset() {
  _inputStream->setAcquireSize(_preferredSize);
  _inputStream->setReleaseSize(_preferredSize);
}

AlgorithmStatus AccumulatorAlgorithm::process() {
  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();

  if (status == OK) {
    consume();
    releaseData();

    return OK;
  }

  // status != SYNC_OK: we couldn't acquire a sufficient number of tokens...

  // if we're not yet at the end of the stream, just return and wait for more
  if (!shouldStop()) return status; // most likely NO_INPUT


  // we are at the end of the stream, we need to work with what's available
  int available = _inputStream->available();
  EXEC_DEBUG("EOS; there are " << available << " available tokens left");

  if (available > 0) {
    _inputStream->setAcquireSize(available);
    _inputStream->setReleaseSize(available);

    status = acquireData();
    if (status != OK) {
      throw EssentiaException("Accumulator EOS internal scheduling error...");
    }

    // consume our very last tokens
    consume();
    releaseData();
  }

  // and now the big moment we've been waiting for all the time! All the tokens
  // from the input stream have been consumed, it is time to output our final result
  finalProduce();

  return FINISHED; // yes we produced something, and we're done!
}


void AccumulatorAlgorithm::declareInputStream(SinkBase& sink, const string& name, const string& desc,
                                              int preferredAcquireSize) {
  _preferredSize = preferredAcquireSize;
  _inputStream = &sink;

  Algorithm::declareInput(sink, preferredAcquireSize, name, desc);
}

void AccumulatorAlgorithm::declareOutputResult(SourceBase& source, const string& name, const string& desc) {
  Algorithm::declareOutput(source, 0, name, desc);
}


} // namespace streaming
} // namespace essentia
