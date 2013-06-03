/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "mintototal.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* MinToTotal::name = "MinToTotal";
const char* MinToTotal::description = DOC("This algorithm computes the ratio between the index of the minimum value of the envelope of a signal and the total length of the envelope.\n"
"An exception is thrown if the input envelop is empty.");

void MinToTotal::compute() {

  const vector<Real>& envelope = _envelope.get();
  Real& minToTotal = _minToTotal.get();

  if (envelope.empty()) {
    throw EssentiaException("MinToTotal: envelope is empty, minToTotal is not defined for an empty envelope");
  }

  minToTotal = Real(argmin(envelope)) / envelope.size();
}

} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {

const char* MinToTotal::name = essentia::standard::MinToTotal::name;
const char* MinToTotal::description = essentia::standard::MinToTotal::description;


void MinToTotal::consume() {
  const vector<Real>& envelope = *((const vector<Real>*)_envelope.getTokens());

  int minIdx = argmin(envelope);
  if (envelope[minIdx] < _min) {
    _min = envelope[minIdx];
    _minIdx = minIdx + _size;
  }

  _size += envelope.size();
}

void MinToTotal::finalProduce() {
  if (_size == 0) {
    throw EssentiaException("MinToTotal: envelope is empty, minToTotal is not defined for an empty envelope");
  }

  _minToTotal.push(Real(_minIdx) / _size);
}

void MinToTotal::reset() {
  AccumulatorAlgorithm::reset();
  _size = 0;
  _min = std::numeric_limits<Real>::max();
  _minIdx = 0;
}

} // namespace streaming
} // namespace essentia
