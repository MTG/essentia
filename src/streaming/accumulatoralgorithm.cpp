/*
 * Copyright (C) 2006-2010 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
