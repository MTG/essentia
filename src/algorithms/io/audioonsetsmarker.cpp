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

#include "audioonsetsmarker.h"
#include "essentiamath.h"


using namespace std;

namespace essentia {
namespace standard {

const char* AudioOnsetsMarker::name = "AudioOnsetsMarker";
const char* AudioOnsetsMarker::description = DOC("This algorithm creates a wave file in which a given audio signal is mixed with a series of time onsets. The sonification of the onsets can be heard as beeps, or as short white noise pulses if configured to do so.\n"
"\n"
"This algorithm will throw an exception if parameter \"filename\" is not supplied");

void AudioOnsetsMarker::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _beep = parameter("type").toString() == "beep";
  _onsets = parameter("onsets").toVectorReal();
  if(!_onsets.empty() && _onsets[0] < 0) {
      throw EssentiaException("AudioOnsetsMarker: onsets cannot be negative");
  }
  for (int i=0; i<(int)(_onsets.size()-1); i++) {
    if (_onsets[i] >= _onsets[i+1]) {
      ostringstream msg;
      msg << "AudioOnsetsMarker: list of onsets not in ascending order: " << _onsets[i] << " >= " << _onsets[i+1];
      throw EssentiaException(msg);
    }
  }
}

void AudioOnsetsMarker::compute() {

  const vector<Real>& input = _input.get();
  vector<Real>& output = _output.get();

  output.resize(input.size(), Real(0.0));
  int burstSize = int(0.04 * _sampleRate);
  for (int i=0; i<int(_onsets.size()); ++i) {
    int start = int(_onsets[i] * _sampleRate);
    int stop = start + burstSize;

    for (int j=start; j<=stop && j< int(input.size()); j++) {
      Real amplitude = 1.0 - Real(j - start) / Real(burstSize);
      amplitude *= 0.5;
      if (_beep) { // mark with a beep
        if (((j-start)/20%2) == 0) output[j] = 0.9*amplitude;
        else output[j] = -0.9*amplitude;
      }
      else {
        output[j] = (rand() / Real(RAND_MAX) * 2.0 - 1.0) * amplitude;
      }
    }
  }

  for (int i=0; i<int(output.size()); ++i) {
    output[i] = (input[i] + output[i]) * 0.5;
  }
}

} // namespace essentia
} // namespace standard


namespace essentia {
namespace streaming {

const char* AudioOnsetsMarker::name = standard::AudioOnsetsMarker::name;
const char* AudioOnsetsMarker::description = standard::AudioOnsetsMarker::description;


AudioOnsetsMarker::AudioOnsetsMarker() : Algorithm(),  _beep(false) {
  _preferredSize = 4096;
  _onsetIdx = 0;
  _processedSamples = 0;
  _burstIdx = 0;
  declareInput(_input, _preferredSize, "signal", "the input signal");
  declareOutput(_output, _preferredSize, "signal", "the input signal mixed with bursts at onset locations");
}

void AudioOnsetsMarker::configure()
{
  _sampleRate = parameter("sampleRate").toReal();
  _beep = parameter("type").toString() == "beep";
  _onsets = parameter("onsets").toVectorReal();

  if(!_onsets.empty() && _onsets[0] < 0) {
      throw EssentiaException("AudioOnsetsMarker: onsets cannot be negative");
  }

  for (int i=0; i<(int)_onsets.size()-1; i++) {
    if (_onsets[i] >= _onsets[i+1]) {
      throw EssentiaException("AudioOnsetsMarker: list of onsets not in ascending order: ", _onsets[i], " >= ", _onsets[i+1]);
    }
    _onsets[i] = int(_onsets[i]*_sampleRate); // cast to ints so it yields same results as standard version
  }
  if (!_onsets.empty()) _onsets[_onsets.size()-1] = int(_onsets[_onsets.size()-1]*_sampleRate);

  // create noise/tone signal
  _burst.resize(int(0.04*_sampleRate));
  for (int i=0; i<int(_burst.size()); i++) {
    Real amplitude = 1.0 - Real(i) / Real(_burst.size());
    amplitude *= 0.5;
    if (_beep) {
        if ((i/20%2) == 0) _burst[i] = 0.9*amplitude;
        else _burst[i] = -0.9*amplitude;
    }
    else {
      _burst[i] = (rand() / Real(RAND_MAX) * 2.0 - 1.0) * amplitude;
    }
  }
}

AlgorithmStatus AudioOnsetsMarker::process() {
  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired");

  if (status != OK) {
    if (!shouldStop()) return status;

    int available = input("signal").available();
    if (available == 0) return FINISHED;

    // otherwise, there's still some audio we can gobble up
    input("signal").setAcquireSize(available);
    input("signal").setReleaseSize(available);
    output("signal").setAcquireSize(available);
    output("signal").setReleaseSize(available);

    return CONTINUE;
  }

  const vector<Real>& input = _input.tokens();
  vector<Real>& output = _output.tokens();

  assert(output.size() == input.size());

  Real out = 0;
  for (int i=0; i<int(input.size()); i++) {
    if (_onsetIdx >= (int)_onsets.size()) {
      out = 0.5*input[i];
    }
    else {
      if (_processedSamples >= int(_onsets[_onsetIdx]) &&
          _processedSamples <= int(_onsets[_onsetIdx] + _burst.size())) {
        // check whether next onset is closer than noise.size():
        if (_onsetIdx < (int)_onsets.size()-1) {
          if (_processedSamples == int(_onsets[_onsetIdx+1])) {
            _burstIdx = 0;
            //cout << "onsetsmarker processing onset: " << _onsetIdx << "/" << _onsets.size() << " at " << _onsets[_onsetIdx]/_sampleRate  << " processedSamples: " << _processedSamples/_sampleRate << endl;
            _onsetIdx++;
            while (_onsetIdx < (int)_onsets.size() && _processedSamples > _onsets[_onsetIdx]) {
              _onsetIdx++;
            }
          }
        }
        out = 0.5*(input[i]+_burst[_burstIdx++]);
        if (_burstIdx >= (int)_burst.size()) {
          _burstIdx = 0;
          //cout << "onsetsmarker processing onset: " << _onsetIdx << "/" << _onsets.size() << " at " << _onsets[_onsetIdx]/_sampleRate  << " processedSamples: " << _processedSamples/_sampleRate << endl;
          _onsetIdx++;
          while (_onsetIdx < (int)_onsets.size() && _processedSamples > _onsets[_onsetIdx]) {
            _onsetIdx++;
          }
        }
      }
      else {
        out = 0.5*input[i];
      }
    }
    output[i] = out;
    _processedSamples++;
    //cout << "onsetIdx: " << _onsetIdx << " pos: " << _onsets[_onsetIdx] << "onsetsSize: " << _onsets.size() <<  " processedSamples: " << _processedSamples << endl;
  }

  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");

  return OK;
}

void AudioOnsetsMarker::reset() {
  Algorithm::reset();
  _onsetIdx = 0;
  _burstIdx = 0;
  _processedSamples = 0;

  _input.setAcquireSize(_preferredSize);
  _input.setReleaseSize(_preferredSize);
  _output.setAcquireSize(_preferredSize);
  _output.setReleaseSize(_preferredSize);
}

} // namespace streaming
} // namespace essentia
