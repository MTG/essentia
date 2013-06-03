/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "leq.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* Leq::name = "Leq";
const char* Leq::description = DOC("This algorithm computes the Equivalent sound level (Leq) of an audio signal. The Leq measure can be derived from the Revised Low-frequency B-weighting (RLB) or from the raw signal as described in [1]. If the signal contains no energy, Leq defaults to essentias definition of silence which is -90dB.\n"
"This algorithm will throw an exception on empty input.\n"
"\n"
"References:\n"
"  [1] Soulodre, G. A., Evaluation of Objective Loudness Meters,\n"
"      Proceedings of the AES 116th Convention, Berlin, Germany, 2004");

void Leq::compute() {

  const std::vector<Real>& signal = _signal.get();
  Real& leq = _leq.get();

  if (signal.empty()) {
    throw EssentiaException("Leq: input signal is empty");
  }

  leq = pow2db(instantPower(signal));
}

} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {

const char* Leq::name = standard::Leq::name;
const char* Leq::description = standard::Leq::description;

void Leq::reset() {
  AccumulatorAlgorithm::reset();
  _size = 0;
  _energy = 0;
}

void Leq::consume() {
  const vector<Real>& signal = *((const vector<Real>*)_signal.getTokens());

  _energy += energy(signal);
  _size += signal.size();
}

void Leq::finalProduce() {
  if (_size == 0) throw EssentiaException("Leq: signal is empty");

  _leq.push(pow2db(_energy/_size));
}


} // namespace streaming
} // namespace essentia
