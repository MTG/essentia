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

#include "loudnessebur128.h"
#include <algorithm> // sort
#include <climits> // DEBUG

using namespace std;

#include "poolstorage.h"

namespace essentia {
namespace streaming {

const char* LoudnessEBUR128::name = "LoudnessEBUR128";
const char* LoudnessEBUR128::description = DOC("This algorithm computes loudness descriptors in accordance with EBU R128 recommendation (TODO refs to both specifications of filter and loundess computation algorithm).\n"
"\n"
"References:\n"
"  [1] EBU Tech 3341-2011. \"Loudness Metering: 'EBU Mode' metering to supplement\n"
"  loudness normalisation in accordance with EBU R 128\"\n"
"  [2] EBU Tech Doc 3342-2011. \"Loudness Range: A measure to supplement loudness\n"
"  normalisation in accordance with EBU R 128\"\n"
"  [3] http://tech.ebu.ch/loudness\n"
);

// TODO: text to reuse in the description

// The measure ‘Loudness Range’ quantifies the variation in a time-varying loudness measurement; it measures the variation of loudness on a macroscopic timescale. Loudness Range is supplementary to the measure of overall loudness, that is, ‘integrated loudness’. The computation of Loudness Range is based on a measurement of loudness level, as specified in ITU-  BS.1770. The term ‘Loudness Range’ is abbreviated ‘LRA’. LRA is measured in units of ‘LU’. It is noted that 1 LU is equivalent to 1 dB. 



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
  _computeIntegrated          = factory.create("UnaryOperatorStream");

  declareInput(_signal, "signal", "the input stereo audio signal");
  declareOutput(_momentaryLoudness, "momentaryLoudness", "momentary loudness (computed by integrating the sum of powers over a sliding rectangular window of 400 ms)");
  declareOutput(_shortTermLoudness, "shortTermLoudness", "short term loudness (computed by integrating the sum of powers over a sliding rectangular window of 3 seconds)");
  declareOutput(_integratedLoudness, "integratedLoudness", "integrated loudness (a loudness value averaged over an arbitrary long time interval)");
  declareOutput(_loudnessRange, "loudnessRange", "loudness range over an arbitrary long time interval [2]");
  declareOutput(_momentaryLoudnessMax, "momentaryLoudnessMax", "observed maximum value for momemtary loudness");
  declareOutput(_momentaryLoudnessMax, "shortTermLoudnessMax", "observed maximum value for short term loudness");


  // Connect input proxy
  _signal >> _loudnessEBUR128Filter->input("signal");

  _loudnessEBUR128Filter->output("signal") >> _frameCutterMomentary->input("signal");
  _loudnessEBUR128Filter->output("signal") >> _frameCutterShortTerm->input("signal");

  // _loudnessEBUR128Filter outputs squared signal
  // according to the specification: filtered signal power = (integral on 0-->T signal² dt) / T
  // therefore, signal power is mean of squared signal
  _frameCutterMomentary->output("frame") >> _meanMomentary->input("array");
  _frameCutterShortTerm->output("frame") >> _meanShortTerm->input("array");

  _meanMomentary->output("mean") >> _computeMomentary->input("array");
  _meanShortTerm->output("mean") >> _computeShortTerm->input("array");

  // Connect output proxies
  _computeMomentary->output("array") >> _momentaryLoudness;
  _computeShortTerm->output("array") >> _shortTermLoudness;


  // TODO: implement "live meter" mode once it is necessary for out tasks
  // for now, gather all values to pool and compute integrated loudness in 
  // the post-processing step
  
  // In a live meter the integrated loudness has to be recalculated from the 
  // preceding (stored) loudness levels of the blocks from the time the 
  // measurement was started, by recalculating the threshold, then applying
  // it to the stored values, every time the meter reading is updated. 
  

  // The integrated loudness uses gating as described in ITU-R BS.1770-2. 
  // The update rate for ‘live meters’ shall be at least 1 Hz. 

  // NOTE: frame size for integrated loudness is the same as for momentary, 
  // however, a fixed hop size of 75% (100ms) is required, which can differ from
  // the user-specified hop size for momentary loudness. Therefore, we can:
  // a) reuse the momentary loudness frame-cutter (faster, fixed hop size) 
  // b) add another frame cutter (slower, flexible hop size)
  _loudnessEBUR128Filter->output("signal")  >> _frameCutterIntegrated->input("signal");
  _frameCutterIntegrated->output("frame")   >> _meanIntegrated->input("array");
  _meanIntegrated->output("mean")           >> _computeIntegrated->input("array");
  _computeIntegrated->output("array")       >> PC(_pool, "integrated_loudness");
  // TODO: we don't need to compute logs for each block, instead comprare block 
  // energies: http://www.hydrogenaud.io/forums/index.php?showtopic=85978&st=50&p=738801&#entry738801

  // NOTE: frame size for loudness range is equal to short-term loudness (3 secs)
  // Hop size is allowed to be implementation dependent, with a minimum block 
  // overlap of 66%, i.e., 2 secs. Therefore, we reuse short-term loudness values.
  _computeShortTerm->output("array")  >> PC(_pool, "shortterm_loudness");



  // TODO: implement Max streaming algorithm
  //_computeMomentary->output("array") >> _momentaryLoudnessMax;
  _computeShortTerm->output("array") >> _shortTermLoudnessMax;
  _loudnessEBUR128Filter->output("signal") >> _momentaryLoudnessMax;

}

LoudnessEBUR128::~LoudnessEBUR128() {}

void LoudnessEBUR128::configure() {

  Real sampleRate = parameter("sampleRate").toReal();
  _hopSize = int(round(parameter("hopSize").toReal() * sampleRate));

  _loudnessEBUR128Filter->configure("sampleRate", sampleRate);
  
  _frameCutterMomentary->configure("frameSize", int(round(0.4 * sampleRate)), // 400ms
                                   "hopSize", _hopSize,
                                   "startFromZero", true,
                                   "silentFrames", "keep");
  _frameCutterShortTerm->configure("frameSize", int(3 * sampleRate), // 3 seconds
                                   "hopSize", _hopSize,
                                   "startFromZero", true,
                                   "silentFrames", "keep");
  // The measurement input to which the gating threshold is applied is the loudness of the
  // 400 ms blocks with a constant overlap between consecutive gating blocks of 75%. 
  _frameCutterIntegrated->configure("frameSize", int(round(0.4 * sampleRate)),
                                    "hopSize", int(round(0.1 * sampleRate)),
                                    "startFromZero", true,
                                    "silentFrames", "keep");

  // loudness = –0.691 + 10 log_10 (power)
  _computeMomentary->configure("type", "log10",
                               "scale", 10.,
                               "shift", -0.691);
  _computeShortTerm->configure("type", "log10",
                               "scale", 10.,
                               "shift", -0.691);
  _computeIntegrated->configure("type", "log10",
                               "scale", 10.,
                               "shift", -0.691);
}


AlgorithmStatus LoudnessEBUR128::process() {
  if (!shouldStop()) return PASS;

  const vector<Real>& loudnessI = _pool.value<vector<Real> >("integrated_loudness");
  
  // compute gated loudness with absolute thresold: 
  // ignore values below -70 LKFS and computed mean of the rest
  Real sum = 0;
  size_t n=0;
  for (size_t i=0; i<loudnessI.size(); ++i) {
    if (loudnessI[i] >= -70.) {
      sum += loudnessI[i];
      n++;
    }
  }
  // relative threshold = gated loudness - 10 LKFS 
  Real threshold = sum / n - 10.;
  cout << "DEBUG threshold=" << threshold << endl;
  cout << "loudnessI=" << loudnessI << endl;

  // compute gated loudness with relative threshold
  sum = 0;
  n = 0;
  for (size_t i=0; i<loudnessI.size(); ++i) {
    if (loudnessI[i] >= threshold) {
      sum += loudnessI[i];
      n++;
    }
  }
  _integratedLoudness.push(sum / n);


  // Compute loudness range based on short-term loudness
  const vector<Real>& loudnessST = _pool.value<vector<Real> >("shortterm_loudness");

  // compute gated loudness with absolute thresold: 
  // ignore values below -70 LKFS and computed mean of the rest
  sum = 0.;
  n=0;
  for (size_t i=0; i<loudnessST.size(); ++i) {
    if (loudnessST[i] >= -70.) {
      sum += loudnessST[i];
      n++;
    }
  }
  // relative threshold = gated loudness - 20 LKFS 
  threshold = sum / n - 10.;

  // remove values lower than the relative threshold
  vector<Real> loudnessSTGated;
  loudnessSTGated.reserve(loudnessST.size());
  for (size_t i=0; i<loudnessST.size(); ++i) {
    if (loudnessST[i]>=threshold) {
      loudnessSTGated.push_back(loudnessST[i]);
    }
  } 
  // C++11:
  //copy_if(loudnessST.begin(), loudnessST.end(), loudnessSTGated.begin(), 
  //        bind2nd(less<Real>(),threshold)));

  sort(loudnessSTGated.begin(), loudnessSTGated.end());

  // LRA is defined as the difference between the estimates of the 10th and the 95th percentiles of the distribution
  size_t iHigh = (size_t) round(0.95*(loudnessSTGated.size()-1));
  size_t iLow = (size_t) round(0.1*(loudnessSTGated.size()-1));

  _loudnessRange.push(loudnessSTGated[iHigh] - loudnessSTGated[iLow]);

  return FINISHED;
}

void LoudnessEBUR128::reset() {
  AlgorithmComposite::reset();
  _network->reset();
  _pool.remove("shortterm_loudness");
  _pool.remove("integrated_loudness");
}

} // namespace streaming
} // namespace essentia
