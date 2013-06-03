/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "maxtototal.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* MaxToTotal::name = "MaxToTotal";
const char* MaxToTotal::description = DOC("This algorithm computes the ratio between the index of the maximum value of the envelope of a signal and the total length of the envelope. This ratio shows how much the maximum amplitude is off-center. Its value is close to 0 if the maximum is close to the beginning (e.g. Decrescendo or Impulsive sounds), close to 0.5 if it is close to the middle (e.g. Delta sounds) and close to 1 if it is close to the end of the sound (e.g. Crescendo sounds)"
"This algorithm is intended to be fed by the output of the Envelope algorithm\n"
"MaxToTotal will throw an exception if the input envelope is empty.");

void MaxToTotal::compute() {

  const vector<Real>& envelope = _envelope.get();
  Real& maxToTotal = _maxToTotal.get();

  if (envelope.empty()) {
    throw EssentiaException("MaxToTotal: envelope is empty, maxToTotal is not defined for an empty envelope");
  }

  maxToTotal = Real(argmax(envelope)) / envelope.size();
}

} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {

const char* MaxToTotal::name = essentia::standard::MaxToTotal::name;
const char* MaxToTotal::description = essentia::standard::MaxToTotal::description;


void MaxToTotal::consume() {
  const vector<Real>& envelope = *((const vector<Real>*)_envelope.getTokens());

  int maxIdx = argmax(envelope);
  if (envelope[maxIdx] > _max) {
    _max = envelope[maxIdx];
    _maxIdx = maxIdx + _size;
  }

  _size += envelope.size();
}

void MaxToTotal::finalProduce() {
  if (_size == 0) {
    throw EssentiaException("MaxToTotal: envelope is empty, maxToTotal is not defined for an empty envelope");
  }

  _maxToTotal.push(Real(_maxIdx) / _size);
}

void MaxToTotal::reset() {
  AccumulatorAlgorithm::reset();
  _size = 0;
  _max = std::numeric_limits<Real>::min();
  _maxIdx = 0;
}

} // namespace streaming
} // namespace essentia
