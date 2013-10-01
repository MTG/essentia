/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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

#include "dynamiccomplexity.h"
#include "essentiamath.h"
#include <list>

using namespace std;

namespace essentia {
namespace standard {


const char* DynamicComplexity::name = "DynamicComplexity";
const char* DynamicComplexity::description = DOC(
"The dynamic complexity is the average absolute deviation from the global\n"
"loudness level estimate on the dB scale. It is related to the dynamic\n"
"range and to the amount of fluctuation in loudness present in a recording.\n"
"\n"
"Silence at the beginning and at the end of a track are ignored in the\n"
"computation in order not to deteriorate the results.\n\n"
"References:\n"
"  [1] S. Streich, Music complexity: a multi-faceted description of audio\n"
"  content, UPF, Barcelona, Spain, 2007.");


void DynamicComplexity::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _frameSize = int(floor(parameter("frameSize").toReal() * _sampleRate));
}

void DynamicComplexity::compute() {
  const vector<Real>& signal = _signal.get();
  Real& complexity = _complexity.get();
  Real& loudness = _loudness.get();

  if (signal.empty()) {
    complexity = 0;
    loudness = -90;
    return;
  }

  Real c = exp(-1.0/(0.035*_sampleRate));

  // create weight vector
  vector<Real> weight(_frameSize, (Real)0.0);
  Real Vweight = 1.0;
  for (int i=_frameSize-1; i>=0; i--) {
    weight[i] = Vweight;
    Vweight *= c;
  }

  // cheap B-curve loudness compensation
  vector<Real> samps;
  filter(samps, signal);

  // compute energy per frame and apply smearing function
  int framenum = signal.size() / _frameSize;
  Real Vms = 0.0;
  vector<Real> VdB(framenum);

  int nSamples = signal.size();

  // energy
  for (int i=0; i<nSamples; i++) {
    samps[i] = samps[i] * samps[i];
  }

  for (int i=0; i<framenum; i++) {
    Vms = Vweight*Vms + (1-c)*inner_product(weight.begin(), weight.end(),
                                            samps.begin()+i*_frameSize, 0.0);
    VdB[i] = pow2db(Vms); //20 * log10(sqrt(Vms) + 1e-9);
  }

  // erase silence at beginning
  int beginIdx = 0;
  while ((beginIdx < framenum) && (VdB[beginIdx] == -90.0)) beginIdx++;
  VdB.erase(VdB.begin(), VdB.begin() + beginIdx);

  // erase silence at end
  int endIdx = VdB.size() - 1;
  while ((endIdx >= 0) && (VdB[endIdx] == -90.0)) endIdx--;
  if (endIdx == -1) // if we don't do this it crashes in VS8.0
    VdB.clear();
  else
    VdB.erase(VdB.begin()+endIdx+1, VdB.end());

  loudness = 0.0;
  complexity = 0.0;

  if (!VdB.empty()) {
    vector<Real> u(VdB.size());
    for (int i=0; i<int(u.size()); i++) u[i] = pow((Real)0.9, -VdB[i]);
    Real s = accumulate(u.begin(), u.end(), 0.0);
    for (int i=0; i<int(u.size()); i++) u[i] /= s;

    loudness = inner_product(u.begin(), u.end(), VdB.begin(), 0.0);

    for (int i=0; i<int(VdB.size()); ++i) {
      complexity += fabs(VdB[i] - loudness);
    }
    complexity /= VdB.size();
  }
  else { // silent input
    loudness = -90.0;
    complexity = 0.0;
  }

  // normalization
  // loudness levels are limited to -90 dB .. 0 dB so if a signal would be
  // half of the time totally silent and the other half at full scale, we
  // would get a value of 45 as the Dynamic Complexity then a safe
  // normalization would be using 45.

}

void DynamicComplexity::filter(vector<Real>& result, const vector<Real>& input) const {
  static const Real nominator[] = { 0.98595, -0.98595 };
  static const Real denominator[] = { 1.0, -0.9719 };

  result.resize(input.size());

  Real b0 = nominator[0];
  Real b1 = nominator[1];
  Real a1 = denominator[1];
  result[0] = b0 * input[0];
  for (int i=1; i<(int)input.size(); i++) {
    result[i] = b0*input[i] + b1*input[i-1] - a1*result[i-1];
  }
}

} // namespace standard
} // namespace essentia

#include "poolstorage.h"
#include "algorithmfactory.h"

namespace essentia {
namespace streaming {

DynamicComplexity::DynamicComplexity() : AlgorithmComposite() {

  _dynAlgo = standard::AlgorithmFactory::create("DynamicComplexity");
  _poolStorage = new PoolStorage<Real>(&_pool, "internal.signal");

  declareInput(_signal, "signal", "the input audio signal");
  declareOutput(_complexity, 0, "dynamicComplexity", "the dynamic complexity coefficient");
  declareOutput(_loudness, 0, "loudness", "an estimate of the loudness [dB]");

  _signal >> _poolStorage->input("data");
}

void DynamicComplexity::configure() {
  _dynAlgo->configure("sampleRate", parameter("sampleRate").toInt(),
                      "frameSize", parameter("frameSize").toReal());
}

AlgorithmStatus DynamicComplexity::process() {
  if (!shouldStop()) return PASS;

  const vector<Real>& signal = _pool.value<vector<Real> >("internal.signal");
  Real complexity;
  Real loudness;

  _dynAlgo->input("signal").set(signal);
  _dynAlgo->output("dynamicComplexity").set(complexity);
  _dynAlgo->output("loudness").set(loudness);
  _dynAlgo->compute();

  _complexity.push(complexity);
  _loudness.push(loudness);

  return FINISHED;
}

void DynamicComplexity::reset() {
  AlgorithmComposite::reset();
  _dynAlgo->reset();
}


} // namespace streaming
} // namespace essentia
