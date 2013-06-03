/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "duration.h"
using namespace std;

namespace essentia {
namespace standard {

const char* Duration::name = "Duration";
const char* Duration::description = DOC("This algorithm returns the total length of a signal recording in seconds.");

void Duration::compute() {
  const vector<Real>& signal = _signal.get();
  Real& duration = _duration.get();

  duration = signal.size()/parameter("sampleRate").toReal();
}


} // namespace standard
} // namespace streaming


namespace essentia {
namespace streaming {

const char* Duration::name = "Duration";
const char* Duration::description = DOC("This algorithm returns the total length of a signal recording in seconds.");

void Duration::reset() {
  AccumulatorAlgorithm::reset();
  _nsamples = 0;
}

void Duration::consume() {
  const vector<Real>& signal = *((const vector<Real>*)_signal.getTokens());

  _nsamples += signal.size();
}

void Duration::finalProduce() {
  _duration.push((Real)(_nsamples / parameter("sampleRate").toReal()));
}

} // namespace streaming
} // namespace essentia
