/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "loudnessvickers.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* LoudnessVickers::name = "LoudnessVickers";
const char* LoudnessVickers::description = DOC("This algorithm computes Vickers's loudness for a given audio signal. Currently, this algorithm only works for signals with a 44100Hz sampling rate. This algorithm is meant to be given frames of audio as input (not entire audio signals). The algorithm described in the paper performs a weighted average of the loudness value computed for each of the given frames, this step is left as a post processing step and is not performed by this algorithm.\n\n"

"References:\n"
"  [1] Vickers, E., Automatic Long-Term Loudness and Dynamics Matching,\n"
"      Proceedings of the AES 111th Convention, New York, NY, USA, 2001,\n"
"      http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.20.4804&rep=rep1&type=pdf");

void LoudnessVickers::configure() {

  // Vms initialization
  _Vms = 0.0;

  _sampleRate = parameter("sampleRate").toReal();

  vector<Real> b(2, 0.0);
  b[0] = 0.98595;
  b[1] = -0.98595;

  vector<Real> a(2, 0.0);
  a[0] = 1.0;
  a[1] = -0.9719;

  // Note: 0.035 is the time constant given in the paper
  _c = exp(-1.0 / (0.035 * _sampleRate));

  _filtering->configure("numerator", b, "denominator", a);
}

void LoudnessVickers::compute() {

  const vector<Real>& signal = _signal.get();
  Real& loudness = _loudness.get();

  // cheap B-curve loudness compensation
  vector<Real> signalFiltered;
  _filtering->input("signal").set(signal);
  _filtering->output("signal").set(signalFiltered);
  _filtering->compute();

  // create weight vector
  vector<Real> weight(signal.size(), 0.0);
  Real Vweight = 1.0;
  // create energy vector
  vector<Real> signalSquare(signal.size(), 0.0);

  for (int i=signal.size()-1; i>=0; --i) {
    weight[i] = Vweight;
    Vweight *= _c;
    signalSquare[i] = signalFiltered[i] * signalFiltered[i];
  }

  // update Vms
  _Vms = Vweight * _Vms + (1 - _c) * inner_product(weight.begin(), weight.end(), signalSquare.begin(), 0.0);

  // calculate loudness
  loudness = pow2db(_Vms); //10 * log10(_Vms + 1e-9);
}
