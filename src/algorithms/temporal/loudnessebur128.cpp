/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include "loudnessebur128.h"
#include <algorithm> // sort
#include "essentiamath.h"

using namespace std;

#include "poolstorage.h"

namespace essentia {
namespace streaming {

const char* LoudnessEBUR128::name = essentia::standard::LoudnessEBUR128::name;
const char* LoudnessEBUR128::category = essentia::standard::LoudnessEBUR128::category;
const char* LoudnessEBUR128::description = essentia::standard::LoudnessEBUR128::description;


LoudnessEBUR128::LoudnessEBUR128() : AlgorithmComposite() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  _frameCutterMomentary       = factory.create("FrameCutter");
  _frameCutterShortTerm       = factory.create("FrameCutter");
  _frameCutterIntegrated      = factory.create("FrameCutter");
  _loudnessEBUR128Filter      = factory.create("LoudnessEBUR128Filter");
  _meanMomentary              = factory.create("Mean");
  _meanShortTerm              = factory.create("Mean");
  _meanIntegrated             = factory.create("Mean");
  _computeMomentary           = factory.create("UnaryOperatorStream");
  _computeShortTerm           = factory.create("UnaryOperatorStream");

  declareInput(_signal, "signal", "the input stereo audio signal");
  declareOutput(_momentaryLoudness, "momentaryLoudness", "momentary loudness (over 400ms) (LUFS)");
  declareOutput(_shortTermLoudness, "shortTermLoudness", "short-term loudness (over 3 seconds) (LUFS)");
  declareOutput(_integratedLoudness, "integratedLoudness", "integrated loudness (overall) (LUFS)");
  declareOutput(_loudnessRange, "loudnessRange", "loudness range over an arbitrary long time interval [3] (dB, LU)");
  //declareOutput(_momentaryLoudnessMax, "momentaryLoudnessMax", "observed maximum value for momentary loudness");
  //declareOutput(_momentaryLoudnessMax, "shortTermLoudnessMax", "observed maximum value for short term loudness");

  // Connect input proxy
  _signal >> _loudnessEBUR128Filter->input("signal");

  _loudnessEBUR128Filter->output("signal").setBufferType(BufferUsage::forLargeAudioStream);
  
  _loudnessEBUR128Filter->output("signal") >> _frameCutterMomentary->input("signal");
  _loudnessEBUR128Filter->output("signal") >> _frameCutterShortTerm->input("signal");

  // _loudnessEBUR128Filter outputs squared signal
  // according to the specification: filtered signal power = (integral on 0-->T signal² dt) / T
  // therefore, signal power is mean of squared signal
  _frameCutterMomentary->output("frame") >> _meanMomentary->input("array");
  _frameCutterShortTerm->output("frame") >> _meanShortTerm->input("array");

  _meanMomentary->output("mean").setBufferType(BufferUsage::forAudioStream);  
  _meanShortTerm->output("mean").setBufferType(BufferUsage::forAudioStream);

  _meanMomentary->output("mean") >> _computeMomentary->input("array");
  _meanShortTerm->output("mean") >> _computeShortTerm->input("array");

  // Connect output proxies
  _computeMomentary->output("array") >> _momentaryLoudness;
  _computeShortTerm->output("array") >> _shortTermLoudness;

  // NOTE: frame size for integrated loudness is the same as for momentary, 
  // however, a fixed hop size of 75% (100ms) is required, which can differ from
  // the user-specified hop size for momentary loudness. Therefore, we can:
  // a) reuse the momentary loudness frame-cutter (faster, fixed hop size) 
  // b) add another frame cutter (slower, flexible hop size)

  _loudnessEBUR128Filter->output("signal")  >> _frameCutterIntegrated->input("signal");
  _frameCutterIntegrated->output("frame")   >> _meanIntegrated->input("array");
  
  // NOTE: We do not need to store values in decibels in the case of integrated 
  // loudness and dynamic range based on short-term loudness, because we would 
  // have to convert them back to power in order to compute mean values.

  // NOTE: frame size for loudness range is equal to short-term loudness (3 secs)
  // Hop size is allowed to be implementation dependent, with a minimum block 
  // overlap of 66%, i.e., 2 secs. Therefore, we reuse short-term loudness values.

  _meanIntegrated->output("mean") >> PC(_pool, "integrated_power");
  _meanShortTerm->output("mean")  >> PC(_pool, "shortterm_power");

  // TODO: implement Max streaming algorithm
  //_computeMomentary->output("array") >> _momentaryLoudnessMax;
  //_computeShortTerm->output("array") >> _shortTermLoudnessMax;

  // TODO: implement "live meter" mode once it will be necessary for our tasks.
  // For now, gather all values to pool and compute integrated loudness in 
  // the post-processing step.
  
  // In a live meter the integrated loudness has to be recalculated from the 
  // preceding (stored) loudness levels of the blocks from the time the 
  // measurement was started, by recalculating the threshold, then applying
  // it to the stored values, every time the meter reading is updated. 
  // The update rate for "live meters" shall be at least 1 Hz. 

  _network = new scheduler::Network(_loudnessEBUR128Filter);
}

LoudnessEBUR128::~LoudnessEBUR128() {
  delete _network;
}

// According to ITU-R BS.1770-2 paper:  loudness = –0.691 + 10 log_10 (power)
inline Real power2loudness(Real power) {
  return 10 * log10(power) -0.691;
}

inline Real loudness2power(Real loudness) {
  return pow(10, (loudness + 0.691) / 10.);
}


void LoudnessEBUR128::configure() {

  Real sampleRate = parameter("sampleRate").toReal();
  bool startFromZero = !parameter("startAtZero").toBool();


  _hopSize = int(round(parameter("hopSize").toReal() * sampleRate));

  _loudnessEBUR128Filter->configure("sampleRate", sampleRate);
  
  _frameCutterMomentary->configure("frameSize", int(round(0.4 * sampleRate)), // 400ms
                                   "hopSize", _hopSize,
                                   "startFromZero", startFromZero,
                                   "silentFrames", "keep");
  _frameCutterShortTerm->configure("frameSize", int(3 * sampleRate), // 3 seconds
                                   "hopSize", _hopSize,
                                   "startFromZero", startFromZero,
                                   "silentFrames", "keep");

  // The measurement input to which the gating threshold is applied is the loudness of the
  // 400 ms blocks with a constant overlap between consecutive gating blocks of 75%. 
  _frameCutterIntegrated->configure("frameSize", int(round(0.4 * sampleRate)),
                                    "hopSize", int(round(0.1 * sampleRate)), 
                                    "startFromZero", startFromZero,
                                    "silentFrames", "keep");

  _computeMomentary->configure("type", "log10",
                               "scale", 10.,
                               "shift", -0.691);
  _computeShortTerm->configure("type", "log10",
                               "scale", 10.,
                               "shift", -0.691);

  // Convert absolute threshold from dB to power
  _absoluteThreshold = loudness2power(-70.);
}


AlgorithmStatus LoudnessEBUR128::process() {
  if (!shouldStop()) return PASS;

  // NOTE: Memory consumption can be optimized by using histograms of a fixed 
  // size (0.1 dB bins are suggested) instead of storing potentially long vector 
  // of values, as it is implemented now. However, this would lead to a 
  // necessity to compute log value for each value of the vector.

  if (!_pool.contains<vector<Real> >("integrated_power") || !_pool.contains<vector<Real> >("shortterm_power")) {
    // do not push anything in the case of empty signal
    E_WARNING("LoudnessEBUR128: empty input signal");
    return FINISHED;
  }

  const vector<Real>& powerI = _pool.value<vector<Real> >("integrated_power");
  
  // compute gated loudness with absolute threshold: 
  // ignore values below -70 LKFS and computed mean of the rest
  Real sum = 0.;
  size_t n=0;
  for (size_t i=0; i<powerI.size(); ++i) {
    if (powerI[i] >= _absoluteThreshold) {
      sum += powerI[i];
      n++;
    }
  }
  // relative threshold = gated loudness in LKFS - 10 LKFS 
  // 10 dB difference means 10 times less power 
  Real threshold = n ? max(sum / n / 10, _absoluteThreshold) : _absoluteThreshold;

  // compute gated loudness with relative threshold
  sum = 0.;
  n = 0;
  for (size_t i=0; i<powerI.size(); ++i) {
    if (powerI[i] >= threshold) {
      sum += powerI[i];
      n++;
    }
  }
  _integratedLoudness.push(power2loudness(n ? sum / n : _absoluteThreshold));
  
  // Compute loudness range based on short-term loudness
  const vector<Real>& powerST = _pool.value<vector<Real> >("shortterm_power");

  // compute gated loudness with absolute threshold: 
  // ignore values below -70 LKFS and computed mean of the rest
  sum = 0.;
  n=0;
  for (size_t i=0; i<powerST.size(); ++i) {
    if (powerST[i] >= _absoluteThreshold) {
      sum += powerST[i];
      n++;
    }
  }
  // relative threshold = gated loudness - 20 LKFS
  // 20 dB difference means 100 times less power
  threshold = n ? max(sum / n / 100, _absoluteThreshold) : _absoluteThreshold;
  
  // remove values lower than the relative threshold
  vector<Real> powerSTGated;
  powerSTGated.reserve(powerST.size());
  for (size_t i=0; i<powerST.size(); ++i) {
    if (powerST[i]>=threshold) {
      powerSTGated.push_back(powerST[i]);
    }
  } 
  // C++11:
  //copy_if(loudnessST.begin(), loudnessST.end(), loudnessSTGated.begin(), 
  //        bind2nd(less<Real>(),threshold)));

  if (powerSTGated.size()) {

    sort(powerSTGated.begin(), powerSTGated.end());

    // LRA is defined as the difference between the estimates of the 10th and 
    // the 95th percentiles of the distribution
    size_t iHigh = (size_t) round(0.95*(powerSTGated.size()-1));
    size_t iLow = (size_t) round(0.1*(powerSTGated.size()-1));

    _loudnessRange.push(power2loudness(powerSTGated[iHigh]) 
                                     - power2loudness(powerSTGated[iLow]));
  }
  else {
    // Consider the dynamic range value of silence to be zero
    _loudnessRange.push((Real) 0.);
  }

  return FINISHED;
}

void LoudnessEBUR128::reset() {
  AlgorithmComposite::reset();
  _pool.remove("shortterm_power");
  _pool.remove("integrated_power");
}

} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

const char* LoudnessEBUR128::name = "LoudnessEBUR128";
const char* LoudnessEBUR128::category = "Loudness/dynamics";
const char* LoudnessEBUR128::description = DOC("This algorithm computes the EBU R128 loudness descriptors of an audio signal.\n"
"\n"
"- The input stereo signal is preprocessed with a K-weighting filter [2] (see LoudnessEBUR128Filter algorithm), composed of two stages: a shelving filter and a high-pass filter (RLB-weighting curve).\n"
"- Momentary loudness is computed by integrating the sum of powers over a sliding rectangular window of 400 ms. The measurement is not gated.\n"
"- Short-term loudness is computed by integrating the sum of powers over a sliding rectangular window of 3 seconds. The measurement is not gated.\n"
"- Integrated loudness is a loudness value averaged over an arbitrary long time interval with gating of 400 ms blocks with two thresholds [2].\n"
"  - Absolute 'silence' gating threshold at -70 LUFS for the computation of the absolute-gated loudness level.\n"
"  - Relative gating threshold, 10 LU below the absolute-gated loudness level.\n"
"- Loudness range is computed from short-term loudness values. It is defined as the difference between the estimates of the 10th and 95th percentiles of the distribution of the loudness values with applied gating [3].\n"
"  - Absolute 'silence' gating threshold at -70 LUFS for the computation of the absolute-gated loudness level.\n"
"  - Relative gating threshold, -20 LU below the absolute-gated loudness level.\n"
"\n"
"References:\n"
"  [1] EBU Tech 3341-2011. \"Loudness Metering: 'EBU Mode' metering to supplement\n"
"  loudness normalisation in accordance with EBU R 128\"\n\n"
"  [2] ITU-R BS.1770-2. \"Algorithms to measure audio programme loudness and true-peak audio level\"\n\n"
"  [3] EBU Tech Doc 3342-2011. \"Loudness Range: A measure to supplement loudness\n"
"  normalisation in accordance with EBU R 128\"\n\n"
"  [4] http://tech.ebu.ch/loudness\n\n"
"  [5] http://en.wikipedia.org/wiki/LKFS\n"
);


LoudnessEBUR128::LoudnessEBUR128() {
  declareInput(_signal, "signal", "the input stereo audio signal");
  declareOutput(_momentaryLoudness, "momentaryLoudness", "momentary loudness (over 400ms) (LUFS)");
  declareOutput(_shortTermLoudness, "shortTermLoudness", "short-term loudness (over 3 seconds) (LUFS)");
  declareOutput(_integratedLoudness, "integratedLoudness", "integrated loudness (overall) (LUFS)");
  declareOutput(_loudnessRange, "loudnessRange", "loudness range over an arbitrary long time interval [3] (dB, LU)");
  //declareOutput(_momentaryLoudnessMax, "momentaryLoudnessMax", "observed maximum value for momentary loudness");
  //declareOutput(_shortTermLoudnessMax, "shortTermLoudnessMax", "observed maximum value for short term loudness");

  createInnerNetwork();
}

LoudnessEBUR128::~LoudnessEBUR128() {
  delete _network;
}

void LoudnessEBUR128::configure() {
  _loudnessEBUR128->configure(INHERIT("sampleRate"), INHERIT("hopSize"), INHERIT("startAtZero"));
}


void LoudnessEBUR128::createInnerNetwork() {
  _loudnessEBUR128 = streaming::AlgorithmFactory::create("LoudnessEBUR128");
  _vectorInput = new streaming::VectorInput<StereoSample>();

  *_vectorInput  >>  _loudnessEBUR128->input("signal");
  _loudnessEBUR128->output("momentaryLoudness")    >> PC(_pool, "momentaryLoudness");
  _loudnessEBUR128->output("shortTermLoudness")    >> PC(_pool, "shortTermLoudness");
  _loudnessEBUR128->output("integratedLoudness")   >> PC(_pool, "integratedLoudness");
  _loudnessEBUR128->output("loudnessRange")        >> PC(_pool, "loudnessRange");
  //_loudnessEBUR128->output("momentaryLoudnessMax") >> PC(_pool, "momentaryLoudnessMax");
  //_loudnessEBUR128->output("shortTermLoudnessMax") >> PC(_pool, "shortTermLoudnessMax");

  _network = new scheduler::Network(_vectorInput);
}

void LoudnessEBUR128::compute() {
  const vector<StereoSample>& signal = _signal.get();
  if (!signal.size()) {
    throw EssentiaException("LoudnessEBUR128: empty input signal");
  }

  _vectorInput->setVector(&signal);
  _network->run();

  vector<Real>& momentaryLoudness = _momentaryLoudness.get();
  vector<Real>& shortTermLoudness = _shortTermLoudness.get();
  Real& integratedLoudness = _integratedLoudness.get();
  Real& loudnessRange = _loudnessRange.get();
  //vector<Real>& momentaryLoudnessMax = _momentaryLoudnessMax.get();
  //vector<Real>& shortTermLoudnessMax = _shortTermLoudnessMax.get();

  momentaryLoudness = _pool.value<vector<Real> >("momentaryLoudness");
  shortTermLoudness = _pool.value<vector<Real> >("shortTermLoudness");
  integratedLoudness = _pool.value<Real>("integratedLoudness");
  loudnessRange = _pool.value<Real>("loudnessRange");

  //TODO: add *Max outputs in the future when users will ask for them. 
  //Meanwhile, the *Max values can be computed from the loudness outputs.
  //We are not interested to be 100% with EBU R128 requirements unless someone
  //will desire to build a real-time loudness meter for broadcasting 
  //applications.

  //momentaryLoudnessMax = _pool.value<vector<Real> >("momentaryLoudnessMax");
  //shortTermLoudnessMax = _pool.value<vector<Real> >("shortTermLoudnessMax");

  //TODO: should output the final max values instead of all observed max values 
  //      as the analysis goes through frames?
  reset();
}

void LoudnessEBUR128::reset() {
  _network->reset();
  _pool.remove("momentaryLoudness");
  _pool.remove("shortTermLoudness");
  _pool.remove("integratedLoudness");
  _pool.remove("loudnessRange");
  //_pool.remove("momentaryLoudnessMax");
  //_pool.remove("shortTermLoudnessMax");
}

} // namespace standard
} // namespace essentia
