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

#include "replaygain.h"
#include "essentiamath.h"
#include <algorithm> // sort

using namespace std;

namespace essentia {
namespace standard {

const char* ReplayGain::name = "ReplayGain";
const char* ReplayGain::category = "Loudness/dynamics";
const char* ReplayGain::description = DOC("This algorithm computes the Replay Gain loudness value of an audio signal. The algorithm is described in detail in [1]. The value returned is the 'standard' ReplayGain value, not the value with 6dB preamplification as computed by lame, mp3gain, vorbisgain, and all widely used ReplayGain programs.\n"
"\n"
"This algorithm is only defined for input signals which size is larger than 0.05ms, otherwise an exception will be thrown.\n"
"\n"
"As a pre-processing step, the algorithm applies equal-loudness filtering to the input signal. This is always done in the standard mode, but it can be turned off in the streaming mode, which is useful in the case one already has an equal-loudness filtered signal.\n"
"\n"
"References:\n"
"  [1] ReplayGain 1.0 specification, https://wiki.hydrogenaud.io/index.php?title=ReplayGain_1.0_specification\n");


void ReplayGain::configure() {
  int sampleRate = parameter("sampleRate").toInt();

  // window set to 50ms
  _rmsWindowSize = (int)(sampleRate * 0.05);

  _eqloudFilter->configure("sampleRate", sampleRate);
}

void ReplayGain::reset() {
  _eqloudFilter->reset();
}

void ReplayGain::compute() {
  const vector<Real>& signal = _signal.get();
  Real& gain = _gain.get();

  // we do not have enough input data to construct a single frame...
  // return the same value as if it was silence
  if ((int)signal.size() < _rmsWindowSize) {
      throw EssentiaException("ReplayGain: The input size must not be less than 0.05ms");
  }

  // 1. Equal loudness filter
  vector<Real> eqloudSignal;
  _eqloudFilter->input("signal").set(signal);
  _eqloudFilter->output("signal").set(eqloudSignal);
  _eqloudFilter->compute();

  // 2. RMS Energy calculation
  int nFrames = (int)eqloudSignal.size() / _rmsWindowSize;
  vector<Real> rms(nFrames, 0.0);

  for (int i = 0; i < nFrames; i++) {
    Real vrms = 0.0;
    for (int j = i*_rmsWindowSize; j < (i+1)*_rmsWindowSize; j++) {
      vrms += eqloudSignal[j] * eqloudSignal[j];
    }
    vrms /= _rmsWindowSize;

    // Convert value to db
    // Note that vrms is energy and not an amplitude as sqrt is not applied
    rms[i] = pow2db(vrms);
  }

  // 3. Statistical processing, as described in the algorithm, the 5% point is taken to
  // represent the overall loudness of the input audio signal
  sort(rms.begin(), rms.end());
  Real loudness = rms[(int)(0.95*rms.size())];

  // 4. Calibration with reference level
  // file is ref_pink.wav, downloaded on reference site (www.replaygain.org)
  Real ref_loudness = -31.492595672607422;

  // 5. Replay gain
  gain = ref_loudness - loudness;
}

} // namespace standard
} // namespace essentia

#include "poolstorage.h"

namespace essentia {
namespace streaming {

const char* ReplayGain::name = essentia::standard::ReplayGain::name;
const char* ReplayGain::category = essentia::standard::ReplayGain::category;
const char* ReplayGain::description = essentia::standard::ReplayGain::description;

ReplayGain::ReplayGain() : _applyEqloud(false) {
  declareInput(_signal, "signal", "the input signal");
  declareOutput(_gain, 0, "replayGain", "the ReplayGain gain value in dB");

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _eqloud   = factory.create("EqualLoudness");
  _fc       = factory.create("FrameCutter");
  _instantp = factory.create("InstantPower");

  // Do not apply equal-loudness filter by default
  _applyEqloud = false;
  _signal                    >>  _fc->input("signal");
  _fc->output("frame")       >> _instantp->input("array");
  _instantp->output("power") >> PC(_pool, "internal.power");
}


ReplayGain::~ReplayGain() {
  if (!_network) {
    _network->deleteAlgorithms();
    delete _network;
  }
  if (!_applyEqloud) delete _eqloud;
}


void ReplayGain::configure() {
  if (!_network) {
    delete _network;
  } 
  _signal.detach();
  if (_applyEqloud) {
    disconnect(_eqloud->output("signal"), _fc->input("signal"));
  }

  _applyEqloud = parameter("applyEqloud").toBool();
  int sampleRate = parameter("sampleRate").toInt();

  // use a 50ms window
  _fc->configure("silentFrames", "noise", 
                 "startFromZero", true,
                 "frameSize", int(0.05 * sampleRate),
                 "hopSize", int(0.05 * sampleRate));
  
  if (_applyEqloud) {
    _signal                    >> _eqloud->input("signal");
    _eqloud->output("signal")  >> _fc->input("signal");

    _eqloud->configure("sampleRate", sampleRate);
    _network = new scheduler::Network(_eqloud, false);
  }
  else {
    _signal                    >>  _fc->input("signal");    
    _network = new scheduler::Network(_fc, false);
  } 
}


AlgorithmStatus ReplayGain::process() {

  if (!shouldStop()) return PASS;

  // it's our pool, so it doesn't matter that we change the order of the values inside
  vector<Real>& powerValues = const_cast<vector<Real>&>(_pool.value<vector<Real> >("internal.power"));

  // 3. Statistical processing
  sort(powerValues.begin(), powerValues.end());
  int size = powerValues.size();
  Real loudness = pow2db(powerValues[(int)(0.95*size)]);

  // 4. Calibration with reference level
  // file is ref_pink.wav, downloaded on reference site (www.replaygain.org)
  Real ref_loudness = -31.462667465209961062;

  // 5. Replay gain
  _gain.push(ref_loudness - loudness);

  return FINISHED;
}


void ReplayGain::reset() {
  // here, just to be on the safe side, we don't use AlgorithmComposite::reset(),
  // because we could then reconfigure with eqloud = true, and then we would have
  // _eqloud as the start of our inner network which hasn't been reset...
  Algorithm::reset();

  _eqloud->reset();
  _fc->reset();
  _instantp->reset();

  _pool.clear();
}


} // namespace streaming
} // namespace essentia
