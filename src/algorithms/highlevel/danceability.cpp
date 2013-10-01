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

#include "danceability.h"

using namespace std;
namespace essentia {
namespace standard {

const char* Danceability::name = "Danceability";
const char* Danceability::description = DOC(
"Calculates the danceability vector for a given signal. The algorithm is\n"
"derived from Detrended Fluctuation Analysis (DFA) described in [1]. The\n"
"parameters minTau and maxTau are used to define the range of time over\n"
"which DFA will be performed. The output of this algorithm is the\n"
"danceability of the audio signal. These values usually range from 0 to 3\n"
"(higher values meaning more danceable).\n"
"Exception is thrown when minTau is greater than maxTau.\n"
"References:\n"
"  [1] Streich, S. and Herrera, P., Detrended Fluctuation Analysis of Music\n"
"  Signals: Danceability Estimation and further Semantic Characterization,\n"
"  Proceedings of the AES 118th Convention, Barcelona, Spain, 2005");

Real Danceability::stddev(const vector<Real>& array, int start, int end) const {

  Real mean_array = mean(array, start, end);
  Real var = 0.0;

  for (int i=start; i<end; i++) {
    Real d = array[i] - mean_array;
    var += d*d;
  }

  return sqrt(var / (end - start - 1.0));
}

void Danceability::configure() {

  Real minTau = parameter("minTau").toReal();
  Real maxTau = parameter("maxTau").toReal();
  Real tauIncrement = parameter("tauMultiplier").toReal();

  if (minTau > maxTau) {
    throw EssentiaException("Danceability: minTau cannot be larger than maximumTauInMs");
  }

  // tau is the number of blocks of 10ms we calculate each time
  _tau.clear();
  for (Real tau = minTau; tau <= maxTau; tau *= tauIncrement) {
    _tau.push_back(int(tau / 10.0));
  }
}

void Danceability::compute() {

  const vector<Real>& signal = _signal.get();
  Real& danceability = _danceability.get();
  Real sampleRate = parameter("sampleRate").toReal();

  //---------------------------------------------------------------------
  // preprocessing:
  // cut up into 10 ms frames and calculate the stddev for each slice
  // store in s(n), then integrate

  int numSamples = signal.size();
  int frameSize = int(0.01 * sampleRate); // 10ms
  int numFrames = numSamples / frameSize;

  vector<Real> s(numFrames, 0.0);

  for (int i=0; i<numFrames; i++) {
    int frameBegin = i * frameSize;
    int frameEnd = min((i+1) * frameSize, numSamples);

    s[i] = stddev(signal, frameBegin, frameEnd);
  }

  // subtract the mean from the array to make it have 0 DC
  Real mean_s = mean(s,0,s.size());
  for (int i=0; i<numFrames; i++)
    s[i] -= mean_s;

  // integrate the signal
  for (int i=1; i<(int)s.size(); i++)
    s[i] += s[i-1];

  //---------------------------------------------------------------------
  // processing

  vector<Real> F(_tau.size(), 0.0);

  int nFValues = 0;

  // for each tau (i.e. block size)
  for (int i=0; i<(int)_tau.size(); i++) {

    int tau = _tau[i];

    // the original algorithm slides/jumps the blocks forward with
    // one sample, but that's very CPU intensive.
    // So, ... for large tau values, lets take larger jumps
    int jump = max(tau/50, 1);

    // perhaps we're working on a short file, then we don't have all values...
    if(numFrames >= tau)
    {
      // cut up the audio in tau-sized blocks
      for(int k=0; k<numFrames - tau; k += jump)
      {
        int frameBegin = k;
        int frameEnd = k + tau;

        // find the average residual error in this block
        // the residual error is sum( squared( signal - linear_regression ) )
        F[i] += residualError(s, frameBegin, frameEnd);
      }

      // the square root of the total residual error, for all the
      // blocks of size tau
      if (numFrames == tau) {
         F[i] = 0.0;
      }
      else {
         F[i] = sqrt(F[i] / ((Real)(numFrames - tau)/(Real)jump));
      }

      nFValues++;
    }
    else
    {
      break;
    }
  }

  danceability = 0.0;

  // the original article tells us: for small tau's we need to adjust alpha (danceability)
  for (int i=0; i<nFValues - 1; i++) {
    if (F[i+1] != 0.0) {
      danceability += log10(F[i+1] / F[i]) / log10( ((Real)_tau[i+1]+3.0) / ((Real)_tau[i]+3.0));
    }
    else {
      danceability = 0.0;
      return;
    }
  }

  danceability /= (nFValues-1);

  if (danceability != 0.0) {
     danceability = 1.0 / danceability;
  }
  else {
     danceability = 0.0;
  }
}

} // namespace standard
} // namespace essentia

#include "poolstorage.h"
#include "algorithmfactory.h"

namespace essentia {
namespace streaming {

const char* Danceability::name = standard::Danceability::name;
const char* Danceability::description = standard::Danceability::description;


Danceability::Danceability() : AlgorithmComposite() {

  _danceabilityAlgo = standard::AlgorithmFactory::create("Danceability");
  _poolStorage = new PoolStorage<Real>(&_pool, "internal.signal");

  declareInput(_signal, 1, "signal", "the input signal");
  declareOutput(_danceability, 0, "danceability", "the danceability value. Normal values range from 0 to ~3. The higher, the more danceable.");

  _signal >> _poolStorage->input("data"); // attach input proxy
}


Danceability::~Danceability() {
  delete _danceabilityAlgo;
  delete _poolStorage;
}

void Danceability::reset() {
  AlgorithmComposite::reset();
  _poolStorage->reset();
}


AlgorithmStatus Danceability::process() {
  if (!shouldStop()) return PASS;

  Real danceability;
  
  _danceabilityAlgo->input("signal").set(_pool.value<vector<Real> >("internal.signal"));
  _danceabilityAlgo->output("danceability").set(danceability);
  _danceabilityAlgo->compute();
  
  _danceability.push(danceability);
  return FINISHED;
}

} // namespace streaming
} // namespace essentia
